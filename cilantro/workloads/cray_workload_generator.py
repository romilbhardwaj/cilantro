from typing import List, Union, Dict

from kubernetes.client import V1Deployment, V1Service, V1ServicePort

from cilantro.workloads.base_workload_generator import BaseWorkloadGenerator
from cilantro.workloads.cray_k8s_utils import get_cray_client_template_deployment, get_cray_server_template_deployment, \
    get_cray_head_template_deployment, get_cray_head_template_service


class CRayWorkloadGenerator(BaseWorkloadGenerator):
    """
    This is a workload generator for the CRay (cilantro-ray) workloads running on ray.
    This workload contains a suite of workloads on Ray - sleep tasks, model serving, db queries
    and ML training. See implementation at https://github.com/romilbhardwaj/cilantro-workloads
    """

    def __init__(self, cluster_type=None, cilantro_image=None):
        """ Constructor. """
        self._cluster_type = cluster_type
        if cilantro_image:
            self._cilantro_image = cilantro_image
        elif cluster_type == 'eks':
            self._cilantro_image = "public.ecr.aws/cilantro/cilantro:latest"
        elif cluster_type == 'kind':
            self._cilantro_image = "public.ecr.aws/cilantro/cilantro:latest"
        else:
            raise ValueError('Invalid input for cluster_type(%s) and/or container_image(%s)' % (
                str(cluster_type), str(cilantro_image)))
        super().__init__()

    def generate_workload_server_objects(self,
                                         app_name: str,
                                         *args,
                                         **kwargs
                                         ) -> List[Union[V1Deployment, V1Service]]:
        # Create the ray cluster head and it's service for incoming connections
        head_deployment = get_cray_head_template_deployment(app_name, *args, **kwargs)
        head_svc = get_cray_head_template_service(app_name)


        # This is the actual workload that must be scaled.
        server_deployment = get_cray_server_template_deployment(app_name,
                                                                head_svc.metadata.name,
                                                                *args,
                                                                **kwargs)

        return [head_svc, head_deployment, server_deployment]

    def generate_workload_client_objects(self,
                                         app_name: str,
                                         *args,
                                         cilantro_client_override_args: Dict[str, str] = None,
                                         cray_client_override_args: Dict[str, str] = None,
                                         head_svc: str = None,
                                         **kwargs) -> List[Union[V1Deployment, V1Service]]:
        """
        Generates workload client objects. The cilantro client is embedded in the same pod.
        :param cilantro_client_override_args: Args to override for cilantro clients
        :param cray_client_override_args: Args to override for cray clients
        :param app_name: Name of the app in the heirarchy.
        :param head_svc: Service name of the ray head servers. If not specified, infers it from app_name
        :param kwargs: other kwargs to pass
        :return:
        """
        cilantro_client_override_args = cilantro_client_override_args if cilantro_client_override_args else {}
        cray_client_override_args = cray_client_override_args if cray_client_override_args else {}
        if head_svc is None:
            # If not specified, infer from app_name
            head_svc = app_name + "-head-svc"

        # ========= Generate defaults =============
        DEFAULT_CRAY_CLIENT_CMD = ["python", "/cray_workloads/cray_workloads/drivers/cray_runscript.py"]
        DEFAULT_CRAY_CLIENT_ARGS = {
            "--cray-utilfreq": "10",
            "--cray-logdir": "/cilantrologs",
            "--cray-workload-type": "sleep_task",
            "--ray-svc-name": head_svc,
            "--sleep-time": "0.2",
            }

        DEFAULT_CILANTRO_CLIENT_CMD = ["python", "/cilantro/cilantro_clients/drivers/cray_to_grpc_driver.py"]
        DEFAULT_CILANTRO_CLIENT_ARGS = {"--log-folder-path": "/cilantrologs",
                                        "--grpc-port": "$(CILANTRO_SERVICE_SERVICE_PORT)",
                                        "--grpc-ip": "$(CILANTRO_SERVICE_SERVICE_HOST)",
                                        "--grpc-client-id": app_name,
                                        "--poll-frequency": "1",
                                        "--slo-type": "latency",
                                        "--slo-latency": 1}

        # =========== Update defaults with workloadinfo args
        cray_args = DEFAULT_CRAY_CLIENT_ARGS.copy()
        cray_args.update(cray_client_override_args)

        cilantro_args = DEFAULT_CILANTRO_CLIENT_ARGS.copy()
        cilantro_args.update(cilantro_client_override_args)

        client_dep = get_cray_client_template_deployment(app_name,
                                                         cray_client_cmd=DEFAULT_CRAY_CLIENT_CMD,
                                                         cray_client_args=cray_args,
                                                         cilantro_client_cmd=DEFAULT_CILANTRO_CLIENT_CMD,
                                                         cilantro_client_args=cilantro_args,
                                                         *args,
                                                         **kwargs)
        return [client_dep]

    def generate_cilantro_client_objects(self,
                                         app_name: str,
                                         *args,
                                         **kwargs) -> List[Union[V1Deployment, V1Service]]:
        return []  # No cilantro clients for this workload since it is co-located with the workload client
