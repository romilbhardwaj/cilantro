from typing import List, Dict, TypeVar

from cilantro.workloads.k8s_utils import get_template_deployment, get_template_service
from kubernetes.client import V1Deployment, V1Service, V1ObjectMeta, V1DeploymentSpec, V1PodTemplateSpec, \
    V1LabelSelector, V1PodSpec, V1Container, V1ContainerPort, V1ServiceSpec, V1ServicePort, V1SecurityContext, \
    V1Capabilities, V1EnvVar, V1EnvVarSource, V1ObjectFieldSelector, V1Lifecycle, V1Handler, V1ExecAction, \
    V1VolumeMount, V1Volume, V1EmptyDirVolumeSource, V1Probe

CASSANDRA_PORTS = {
    "intra-node": 7000,
    "tls-intra-node": 7001,
    "jmx": 7199,
    "cql": 9042,
    "thrift": 9160
}

CILANTRO_PORTS = {
    'grpc': 10000
}

# These must be strs
DEFAULT_YCSB_CLIENT_THREADCOUNT = "16"
DEFAULT_YCSB_CLIENT_RECORDCOUNT = "1000"
DEFAULT_YCSB_CLIENT_OPERATIONCOUNT = "5000"
DEFAULT_YCSP_CLIENT_TARGET_THROUGHPUT = "1000000000"

cassandra_readiness_probe = V1Probe(_exec=V1ExecAction(command=["/bin/bash", "-c", "/scripts/ready-probe.sh"]),
                                    initial_delay_seconds=10,
                                    timeout_seconds=5,
                                    failure_threshold=10,
                                    period_seconds=10)
# ==========================================================
# =================== SEED OBJECTS =========================
# ==========================================================

def get_cassandra_seed_template_deployment(app_name: str,
                                           *args,
                                           **kwargs) -> V1Deployment:
    """
    Defines the cassandra seed. Does not get autoscaled with more resources - only 1 seed is needed.
    Also not counted as a workload, thus is_workload is false.
    :param app_name: Name of the application in the hierarchy. DO NOT append anything before passing it to this method.
    :return: V1Deployment object
    """
    # Assign defaults:
    app_name = app_name if app_name else "default"
    seed_name = app_name + "-seed"
    is_workload = "false"
    default_replicas = 1
    container_image = "public.ecr.aws/cilantro/data-serving:server"
    container_ports = list(CASSANDRA_PORTS.values())
    container_image_pull_policy = "Always"
    container_command = None
    container_args = None
    cass_envs = [V1EnvVar("MAX_HEAP_SIZE", "512M"),
                 V1EnvVar("HEAP_NEWSIZE", "100M"),
                 V1EnvVar("POD_IP", value_from=V1EnvVarSource(
                     field_ref=V1ObjectFieldSelector(field_path="status.podIP")))]

    override_labels = {f"can-serve-{app_name}": "true"}

    seed_deployment = get_template_deployment(app_name=seed_name,
                                              is_workload=is_workload,
                                              default_replicas=default_replicas,
                                              container_image=container_image,
                                              container_ports=container_ports,
                                              container_image_pull_policy=container_image_pull_policy,
                                              container_command=container_command,
                                              container_args=container_args,
                                              override_labels=override_labels)

    # All updates we make here happen in place - no need to create new V1DeploymentObject
    # Set security context
    sec_context = V1SecurityContext(privileged=True, capabilities=V1Capabilities(add=["IPC_LOCK"]))
    container = seed_deployment.spec.template.spec.containers[0]
    container.security_context = sec_context
    container.readiness_probe = cassandra_readiness_probe

    # Update environment variables
    current_envs = container.env
    if not current_envs:
        container.env = cass_envs
    else:
        current_envs.extend(cass_envs)

    return seed_deployment


def get_cassandra_seed_template_service(app_name: str,
                                        *args,
                                        meta_labels: Dict[str, str] = None,
                                        **kwargs) -> V1Service:
    """
    Defines a headless service that allows for seed discovery.
    :param app_name: Name of the application in the hierarchy. DO NOT append anything before passing it to this method.
    :return: V1Service object
    """
    # Assign defaults:
    app_name = app_name if app_name else "default"
    seed_name = app_name + "-seed"
    meta_labels = meta_labels if meta_labels else {}
    meta_labels["app"] = seed_name
    ports = [V1ServicePort(name=n, port=p) for n, p in CASSANDRA_PORTS.items()]
    selector_match_labels = {"app": seed_name}

    seed_svc = get_template_service(svc_name=seed_name + "-svc",
                                    meta_labels=meta_labels,
                                    ports=ports,
                                    selector_match_labels=selector_match_labels)

    # Make it a "headless" service for cassandra seeds
    seed_svc.spec.cluster_ip = "None"

    return seed_svc


# ==========================================================
# ================= SERVER OBJECTS =========================
# ==========================================================

def get_cassandra_server_template_deployment(app_name: str,
                                             seed_svc_name: str,
                                             *args,
                                             threshold: str = "",
                                             app_weight: str = "",
                                             app_unit_demand: str = "",
                                             **kwargs
                                             ) -> V1Deployment:
    """
    Defines the cassandra server object. This is deployment that gets scaled with more resources.
    :param app_name: Name of the application in the hierarchy. DO NOT append anything before passing it to this method.
    :return: V1Deployment object
    """
    # Assign defaults:
    app_name = app_name if app_name else "default"
    is_workload = "true"
    threshold = threshold if threshold else ""
    app_weight = app_weight if app_weight else app_weight
    app_unit_demand = app_unit_demand if app_unit_demand else ""
    default_replicas = 1
    container_image = "public.ecr.aws/cilantro/data-serving:server"
    container_ports = list(CASSANDRA_PORTS.values())
    container_image_pull_policy = "Always"
    container_command = None
    container_args = None
    server_envs = [V1EnvVar("MAX_HEAP_SIZE", "512M"),
                   V1EnvVar("HEAP_NEWSIZE", "100M"),
                   V1EnvVar("CASSANDRA_SEEDS", seed_svc_name),
                   V1EnvVar("POD_IP", value_from=V1EnvVarSource(
                       field_ref=V1ObjectFieldSelector(field_path="status.podIP")))]

    override_labels = {f"can-serve-{app_name}": "true"}

    server_deployment = get_template_deployment(app_name=app_name,
                                                is_workload=is_workload,
                                                threshold=str(threshold),
                                                app_weight=str(app_weight),
                                                app_unit_demand=str(app_unit_demand),
                                                default_replicas=default_replicas,
                                                container_image=container_image,
                                                container_ports=container_ports,
                                                container_image_pull_policy=container_image_pull_policy,
                                                container_command=container_command,
                                                container_args=container_args,
                                                override_labels=override_labels)

    # All updates we make here happen in place - no need to create new V1DeploymentObject
    # Set security context
    sec_context = V1SecurityContext(privileged=True, capabilities=V1Capabilities(add=["IPC_LOCK"]))
    container = server_deployment.spec.template.spec.containers[0]
    container.security_context = sec_context
    container.readiness_probe = cassandra_readiness_probe

    # Update environment variables
    current_envs = container.env
    if not current_envs:
        container.env = server_envs
    else:
        current_envs.extend(server_envs)

    return server_deployment


def get_cassandra_server_template_service(app_name: str,
                                          *args,
                                          meta_labels: Dict[str, str] = None,
                                          **kwargs) -> V1Service:
    """
    Defines a service for clients to connect to the servers with.
    :param app_name: Name of the application in the hierarchy. DO NOT append anything before passing it to this method.
    :return: V1Service object
    """
    # Assign defaults:
    app_name = app_name if app_name else "default"
    svc_name = app_name + "-svc"
    meta_labels = meta_labels if meta_labels else {}
    meta_labels["app"] = app_name
    ports = [V1ServicePort(name=n, port=p) for n, p in CASSANDRA_PORTS.items()]
    selector_match_labels = {f"can-serve-{app_name}": "true"}

    server_svc = get_template_service(svc_name=svc_name,
                                      meta_labels=meta_labels,
                                      ports=ports,
                                      selector_match_labels=selector_match_labels)
    return server_svc

# ==========================================================
# ================= CLIENT OBJECTS =========================
# ==========================================================

def get_cassandra_client_template_deployment(app_name: str,
                                             server_svc: str,
                                             threshold: float,
                                             *args,
                                             log_dir: str = "/cilantrologs",
                                             cilantro_image: str = "public.ecr.aws/cilantro/cilantro:latest",
                                             cilantro_client_cmd: List[str] = None,
                                             cilantro_client_args: List[str] = None,
                                             env_noinitdb: bool = False,
                                             env_usek8s: bool = False,
                                             threadcount: str = None,
                                             operationcount: str = None,
                                             recordcount: str = None,
                                             **kwargs) -> V1Deployment:
    """
    Defines the deployment for the cassandra client.
    A pod contains two containers - cassandra client and cilantro client.
    Also not counted as a workload, thus is_workload is false.
    :param app_name: Name of the application in the hierarchy. DO NOT append anything before passing it to this method.
    :param server_svc: Service name of the servers
    :param noinitdb: If set to true, the client assumes the databases are initialized. This is true for the horizonal scaling case.
    :param use_k8s: If set to true, the client uses k8s to resolve individual IPs of the servers, instead of k8s load balancing. This is true for the horizonal scaling case.
    :return: V1Deployment object
    """
    # Assign defaults:
    app_name = app_name if app_name else "default"
    client_name = app_name + "-client"
    is_workload = "false"
    default_replicas = 1
    container_image = "public.ecr.aws/cilantro/data-serving:client"
    container_ports = list(CASSANDRA_PORTS.values())
    container_image_pull_policy = "Always"
    container_command = None
    container_args = [f"{server_svc}", f"{log_dir}"]

    threadcount = threadcount if threadcount else DEFAULT_YCSB_CLIENT_THREADCOUNT
    operationcount = operationcount if operationcount else DEFAULT_YCSB_CLIENT_OPERATIONCOUNT
    recordcount = recordcount if recordcount else DEFAULT_YCSB_CLIENT_RECORDCOUNT
    if 'client_target_throughput' in kwargs:
        client_target_throughput = kwargs['client_target_throughput']
    else:
        client_target_throughput = DEFAULT_YCSP_CLIENT_TARGET_THROUGHPUT

    print(threadcount, operationcount, recordcount)
    default_cmd = ["python", "/cilantro/cilantro_clients/drivers/ycsb_to_grpc_driver.py"]
    cilantro_client_cmd = cilantro_client_cmd if cilantro_client_cmd else default_cmd
    default_args = ["--log-folder-path", "/cilantrologs",
                    "--grpc-port", "$(CILANTRO_SERVICE_SERVICE_PORT)",
                    "--grpc-ip", "$(CILANTRO_SERVICE_SERVICE_HOST)",
                    "--grpc-client-id", app_name,
                    "--poll-frequency", "1",
                    "--slo-latency", str(kwargs['slo_latency']),
                    "--app-name", app_name]
    cilantro_client_args = cilantro_client_args if cilantro_client_args else default_args

    cass_envs = [V1EnvVar("MAX_HEAP_SIZE", "512M"),
                 V1EnvVar("HEAP_NEWSIZE", "100M"),
                 V1EnvVar("THREADCOUNT", str(threadcount)),
                 V1EnvVar("RECORDCOUNT", str(recordcount)),
                 V1EnvVar("OPERATIONCOUNT", str(operationcount)),
                 V1EnvVar("TARGET", str(client_target_throughput)),
                 V1EnvVar("POD_IP", value_from=V1EnvVarSource(
                     field_ref=V1ObjectFieldSelector(field_path="status.podIP")))]

    if env_noinitdb:
        cass_envs.append(V1EnvVar("NOINITDB", "1"))

    if env_usek8s:
        cass_envs.append(V1EnvVar("USEK8S", "1"))

    override_labels = {f"can-serve-{app_name}": "false"}

    client_deployment = get_template_deployment(app_name=client_name,
                                              is_workload=is_workload,
                                              default_replicas=default_replicas,
                                              container_image=container_image,
                                              container_ports=container_ports,
                                              container_image_pull_policy=container_image_pull_policy,
                                              container_command=container_command,
                                              container_args=container_args,
                                              override_labels=override_labels)

    # Add volume to the pod spec
    podspec = client_deployment.spec.template.spec
    log_volume = V1Volume(empty_dir=V1EmptyDirVolumeSource(), name="log-share")
    podspec.volumes = [log_volume]
    podspec.termination_grace_period_seconds = 0

    # Create volume mount for shared log mount
    log_mount = V1VolumeMount(mount_path="/cilantrologs", name="log-share")

    # Updates to YCSB container.
    # All updates we make here happen in place - no need to create new V1DeploymentObject
    ycsb_container = podspec.containers[0]
    ycsb_container.name = "ycsb"
    sec_context = V1SecurityContext(privileged=True, capabilities=V1Capabilities(add=["IPC_LOCK"]))
    ycsb_container.security_context = sec_context
    ycsb_container.lifecycle = V1Lifecycle(pre_stop=V1Handler(_exec=V1ExecAction(["/bin/sh", "-c", "nodetool drain"])))
    ycsb_container.volume_mounts = [log_mount]

    current_envs = ycsb_container.env
    if not current_envs:
        ycsb_container.env = cass_envs
    else:
        current_envs.extend(cass_envs)

    # Define Cilantro Client container and add it to list of containers
    cilclient_container = V1Container(name="cilantroclient",
                                      image=cilantro_image,
                                      image_pull_policy="Always",
                                      volume_mounts=[log_mount],
                                      ports=[V1ContainerPort(container_port=p) for p in CILANTRO_PORTS.values()],
                                      command=cilantro_client_cmd,
                                      args=cilantro_client_args
                                      )
    podspec.containers.append(cilclient_container)
    return client_deployment


# ==========================================================
# ======= HORIZONTAL SCALING SERVER OBJECTS ================
# ==========================================================

def get_cassandra_serverhorz_template_deployment(app_name: str,
                                             *args,
                                             threshold: str = "",
                                             app_weight: str = "",
                                             app_unit_demand: str = "",
                                             recordcount: str = None,
                                             **kwargs
                                             ) -> V1Deployment:
    """
    Defines the deployment for the horizontal scaling cassandra deployment
    :param app_name: Name of the application in the hierarchy. DO NOT append anything before passing it to this method.
    :return: V1Deployment object
    """
    # Assign defaults:
    app_name = app_name if app_name else "default"
    is_workload = "true"
    threshold = threshold if threshold else ""
    app_weight = app_weight if app_weight else app_weight
    app_unit_demand = app_unit_demand if app_unit_demand else ""
    default_replicas = 1
    container_image = "public.ecr.aws/cilantro/data-serving:server-horz"
    container_ports = list(CASSANDRA_PORTS.values())
    container_image_pull_policy = "Always"
    container_command = None
    container_args = None
    recordcount = recordcount if recordcount else DEFAULT_YCSB_CLIENT_RECORDCOUNT
    server_envs = [V1EnvVar("MAX_HEAP_SIZE", "512M"),
                   V1EnvVar("HEAP_NEWSIZE", "100M"),
                   V1EnvVar("RECORDCOUNT", str(recordcount)),
                   V1EnvVar("POD_IP", value_from=V1EnvVarSource(
                       field_ref=V1ObjectFieldSelector(field_path="status.podIP")))]

    override_labels = {f"can-serve-{app_name}": "true"}

    server_deployment = get_template_deployment(app_name=app_name,
                                                is_workload=is_workload,
                                                threshold=str(threshold),
                                                app_weight=str(app_weight),
                                                app_unit_demand=str(app_unit_demand),
                                                default_replicas=default_replicas,
                                                container_image=container_image,
                                                container_ports=container_ports,
                                                container_image_pull_policy=container_image_pull_policy,
                                                container_command=container_command,
                                                container_args=container_args,
                                                override_labels=override_labels)

    # All updates we make here happen in place - no need to create new V1DeploymentObject
    # Set security context
    sec_context = V1SecurityContext(privileged=True, capabilities=V1Capabilities(add=["IPC_LOCK"]))
    container = server_deployment.spec.template.spec.containers[0]
    container.security_context = sec_context

    # ================ Readiness Probe ===================
    probe = V1Probe(_exec=V1ExecAction(command=["/bin/bash", "-c", "/scripts/ready-probe.sh"]),
                                        initial_delay_seconds=5,
                                        timeout_seconds=5,
                                        failure_threshold=10,
                                        period_seconds=5)
    container.readiness_probe = probe

    # Update environment variables
    current_envs = container.env
    if not current_envs:
        container.env = server_envs
    else:
        current_envs.extend(server_envs)

    return server_deployment


def get_cassandra_serverhorz_template_service(app_name: str,
                                          *args,
                                          meta_labels: Dict[str, str] = None,
                                          **kwargs) -> V1Service:
    """
    Defines a service for clients to connect to the servers with.
    :param app_name: Name of the application in the hierarchy. DO NOT append anything before passing it to this method.
    :return: V1Service object
    """
    # Assign defaults:
    app_name = app_name if app_name else "default"
    svc_name = app_name + "-svc"
    meta_labels = meta_labels if meta_labels else {}
    meta_labels["app"] = app_name
    ports = [V1ServicePort(name=n, port=p) for n, p in CASSANDRA_PORTS.items()]
    selector_match_labels = {f"can-serve-{app_name}": "true"}

    server_svc = get_template_service(svc_name=svc_name,
                                      meta_labels=meta_labels,
                                      ports=ports,
                                      selector_match_labels=selector_match_labels)

    server_svc.spec.cluster_ip = "None"
    return server_svc
