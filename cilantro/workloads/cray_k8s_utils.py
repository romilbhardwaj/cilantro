from typing import List, Dict, TypeVar

from cilantro.workloads.k8s_utils import get_template_deployment, get_template_service
from kubernetes.client import V1Deployment, V1Service, V1ObjectMeta, V1DeploymentSpec, V1PodTemplateSpec, \
    V1LabelSelector, V1PodSpec, V1Container, V1ContainerPort, V1ServiceSpec, V1ServicePort, V1SecurityContext, \
    V1Capabilities, V1EnvVar, V1EnvVarSource, V1ObjectFieldSelector, V1Lifecycle, V1Handler, V1ExecAction, \
    V1VolumeMount, V1Volume, V1EmptyDirVolumeSource, V1Probe, V1ResourceFieldSelector, V1ResourceRequirements

RAY_HEAD_PORTS = {
    "client": 10001,
    "dashboard": 8265,
    "redis": 6379,
}

CILANTRO_PORTS = {
    'grpc': 10000
}

def convert_dictargs_to_listargs(dict_args: Dict):
    # Converts dictionary args to sequential args for cmd line
    listargs = []
    for k,v in dict_args.items():
        listargs.append(k)
        listargs.append(v)
    return listargs

# ==========================================================
# =================== RAY HEAD OBJECTS =====================
# ==========================================================

def get_cray_head_template_deployment(app_name: str,
                                      *args,
                                      **kwargs) -> V1Deployment:
    """
    Defines the ray head node. Does not get autoscaled with more resources - only 1 head is needed.
    Also not counted as a workload, thus is_workload is false.
    :param app_name: Name of the application in the hierarchy. DO NOT append anything before passing it to this method.
    :return: V1Deployment object
    """
    # Assign defaults:
    app_name = app_name if app_name else "default"
    head_name = app_name + "-head"
    is_workload = "false"
    default_replicas = 1
    container_image = "public.ecr.aws/cilantro/cray-workloads:latest"
    container_ports = list(RAY_HEAD_PORTS.values())
    container_image_pull_policy = "Always"
    container_command = ["/bin/bash", "-c", "--"]
    container_args = ["ray start --head --port=6379 --redis-shard-ports=6380,6381 --num-cpus=0 --object-manager-port=12345 --node-manager-port=12346 --dashboard-host=0.0.0.0 --block"]
    envs = [V1EnvVar("POD_IP", value_from=V1EnvVarSource(
        field_ref=V1ObjectFieldSelector(field_path="status.podIP"))),
            V1EnvVar("MY_CPU_REQUEST", value_from=V1EnvVarSource(
                resource_field_ref=V1ResourceFieldSelector(resource="requests.cpu")))
            ]

    head_deployment = get_template_deployment(app_name=head_name,
                                              is_workload=is_workload,
                                              default_replicas=default_replicas,
                                              container_image=container_image,
                                              container_ports=container_ports,
                                              container_image_pull_policy=container_image_pull_policy,
                                              container_command=container_command,
                                              container_args=container_args)

    # All updates we make here happen in place - no need to create new V1DeploymentObject
    podspec = head_deployment.spec.template.spec

    # Add dshm volume
    dshm_volume = V1Volume(empty_dir=V1EmptyDirVolumeSource(medium="Memory"), name="dshm")
    podspec.volumes = [dshm_volume]
    dshm_mount = V1VolumeMount(mount_path="/dev/shm", name="dshm")

    # Set resource requirements
    resreq = V1ResourceRequirements(requests={"cpu": "100m",
                                              "memory": "512m"})

    container = podspec.containers[0]
    container.volume_mounts = [dshm_mount]
    container.resources = resreq

    # Update environment variables
    current_envs = container.env
    if not current_envs:
        container.env = envs
    else:
        current_envs.extend(envs)

    return head_deployment


def get_cray_head_template_service(app_name: str,
                                   *args,
                                   meta_labels: Dict[str, str] = None,
                                   **kwargs) -> V1Service:
    """
    Defines a headless service that allows for head connections.
    :param app_name: Name of the application in the hierarchy. DO NOT append anything before passing it to this method.
    :return: V1Service object
    """
    # Assign defaults:
    app_name = app_name if app_name else "default"
    head_name = app_name + "-head"
    meta_labels = meta_labels if meta_labels else {}
    meta_labels["app"] = app_name
    ports = [V1ServicePort(name=n, port=p) for n, p in RAY_HEAD_PORTS.items()]
    selector_match_labels = {"app": head_name}

    head_svc = get_template_service(svc_name=head_name + "-svc",
                                    meta_labels=meta_labels,
                                    ports=ports,
                                    selector_match_labels=selector_match_labels)

    # Make it a "headless" service for cassandra
    # head_svc.spec.cluster_ip = "None"

    return head_svc


# ==========================================================
# =========== WORKER (Server) OBJECTS ======================
# ==========================================================

def get_cray_server_template_deployment(app_name: str,
                                        head_svc_name: str,
                                        *args,
                                        **kwargs
                                        ) -> V1Deployment:
    """
    Defines the cray server/workload objects. This is deployment that gets scaled with more resources.
    :param app_name: Name of the application in the hierarchy. DO NOT append anything before passing it to this method.
    :return: V1Deployment object
    """
    # Assign defaults:
    app_name = app_name if app_name else "default"
    is_workload = "true"
    default_replicas = 1
    container_image = "public.ecr.aws/cilantro/cray-workloads:latest"
    container_ports = list(RAY_HEAD_PORTS.values())
    container_image_pull_policy = "Always"
    container_command = ["/bin/bash", "-c", "--"]
    head_svc_envvar_name_prefix = head_svc_name.replace("-", "_").upper()
    container_args = [f"ray start --num-cpus=$MY_CPU_REQUEST --address=${head_svc_envvar_name_prefix}_SERVICE_HOST:${head_svc_envvar_name_prefix}_SERVICE_PORT_REDIS --object-manager-port=12345 --node-manager-port=12346 --block"]
    server_envs = [
        V1EnvVar("POD_IP", value_from=V1EnvVarSource(
            field_ref=V1ObjectFieldSelector(field_path="status.podIP"))),
            V1EnvVar("MY_CPU_REQUEST", value_from=V1EnvVarSource(
                resource_field_ref=V1ResourceFieldSelector(resource="requests.cpu")))
            ]

    server_deployment = get_template_deployment(app_name=app_name,
                                                is_workload=is_workload,
                                                default_replicas=default_replicas,
                                                container_image=container_image,
                                                container_ports=container_ports,
                                                container_image_pull_policy=container_image_pull_policy,
                                                container_command=container_command,
                                                container_args=container_args)

    # All updates we make here happen in place - no need to create new V1DeploymentObject
    podspec = server_deployment.spec.template.spec

    # Add dshm volume
    dshm_volume = V1Volume(empty_dir=V1EmptyDirVolumeSource(medium="Memory"), name="dshm")
    podspec.volumes = [dshm_volume]
    dshm_mount = V1VolumeMount(mount_path="/dev/shm", name="dshm")

    # Set resource requirements
    resreq = V1ResourceRequirements(requests={"cpu": "100m",
                                              "memory": "512m"})

    container = podspec.containers[0]
    container.volume_mounts = [dshm_mount]
    container.resources = resreq

    # Update environment variables
    current_envs = container.env
    if not current_envs:
        container.env = server_envs
    else:
        current_envs.extend(server_envs)

    return server_deployment


# ==========================================================
# ================= CLIENT OBJECTS =========================
# ==========================================================

def get_cray_client_template_deployment(app_name: str,
                                        cray_client_cmd: List[str],
                                        cray_client_args: Dict[str, str],
                                        cilantro_client_cmd: List[str],
                                        cilantro_client_args: Dict[str, str],
                                        *args,
                                        cilantro_image: str = "public.ecr.aws/cilantro/cilantro:latest",
                                        **kwargs) -> V1Deployment:
    """
    Defines the deployment for the cray client.
    A pod contains two containers - cray client and cilantro client.
    Also not counted as a workload, thus is_workload is false.
    :param app_name: Name of the application in the hierarchy. DO NOT append anything before passing it to this method.
    :return: V1Deployment object
    """
    # Assign defaults:
    app_name = app_name if app_name else "default"
    client_name = app_name + "-client"
    is_workload = "false"
    default_replicas = 1
    container_image = "public.ecr.aws/cilantro/cray-workloads:latest"
    container_ports = []
    container_image_pull_policy = "Always"

    envs = [V1EnvVar("POD_IP", value_from=V1EnvVarSource(
        field_ref=V1ObjectFieldSelector(field_path="status.podIP")))]

    cray_listargs = convert_dictargs_to_listargs(cray_client_args)

    client_deployment = get_template_deployment(app_name=client_name,
                                                is_workload=is_workload,
                                                default_replicas=default_replicas,
                                                container_image=container_image,
                                                container_ports=container_ports,
                                                container_image_pull_policy=container_image_pull_policy,
                                                container_command=cray_client_cmd,
                                                container_args=cray_listargs)

    # Add volume to the pod spec
    podspec = client_deployment.spec.template.spec
    log_volume = V1Volume(empty_dir=V1EmptyDirVolumeSource(), name="log-share")
    podspec.volumes = [log_volume]
    podspec.termination_grace_period_seconds = 0

    # Create volume mount for shared log mount
    log_mount = V1VolumeMount(mount_path="/cilantrologs", name="log-share")

    # Updates to YCSB container.
    # All updates we make here happen in place - no need to create new V1DeploymentObject
    crayclient_container = podspec.containers[0]
    crayclient_container.name = "cray"
    crayclient_container.volume_mounts = [log_mount]

    current_envs = crayclient_container.env
    if not current_envs:
        crayclient_container.env = envs
    else:
        current_envs.extend(envs)

    cilantro_listargs = convert_dictargs_to_listargs(cilantro_client_args)

    # Define Cilantro Client container and add it to list of containers
    cilclient_container = V1Container(name="cilantroclient",
                                      image=cilantro_image,
                                      image_pull_policy="Always",
                                      volume_mounts=[log_mount],
                                      ports=[V1ContainerPort(container_port=p) for p in CILANTRO_PORTS.values()],
                                      command=cilantro_client_cmd,
                                      args=cilantro_listargs
                                      )
    podspec.containers.append(cilclient_container)
    return client_deployment
