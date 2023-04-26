from typing import List, Union

from kubernetes.client import V1Deployment, V1Service, V1ServicePort

from cilantro.workloads.base_workload_generator import BaseWorkloadGenerator
from cilantro.workloads.k8s_utils import get_template_deployment, get_template_service


class PropAllocWorkloadGenerator(BaseWorkloadGenerator):
    """
    The PropAlloc workload is used as a test workload.
    Its cilantro client return utility and other metrics as function of the k8s allocation.
    """

    def generate_workload_server_objects(self,
                                         app_name: str,
                                         threshold: float,
                                         app_weight: float,
                                         app_unit_demand: float,
                                         *args,
                                         **kwargs
                                         )-> List[Union[V1Deployment, V1Service]]:
        # We run nginx servers as a dummy workload
        workload_dep = get_template_deployment(app_name=app_name,
                                               is_workload="true",
                                               threshold=str(threshold),
                                               app_weight=str(app_weight),
                                               app_unit_demand=str(app_unit_demand),
                                               container_image="nginx:1.15.4",
                                               container_ports=[80],
                                               **kwargs)
        # workload_service = None
        return [workload_dep]

    def generate_workload_client_objects(self, *args, **kwargs) -> List[Union[V1Deployment, V1Service]]:
        return []   # No workload clients for this workload since its a dummy

    def generate_cilantro_client_objects(self,
                                         app_name:str,
                                         *args,
                                         **kwargs) -> List[Union[V1Deployment, V1Service]]:
        cilantroclient_dep = get_template_deployment(app_name=app_name + "--cc",
                                                     is_workload="false",
                                                     container_image="public.ecr.aws/cilantro/cilantro:latest",
                                                     container_ports=[10000],
                                                     container_command=["python", "/cilantro/cilantro_clients/drivers/k8s_to_grpc_driver.py"],
                                                     container_args=["--k8s-app-name", app_name, "--grpc-port", "$(CILANTRO_SERVICE_SERVICE_PORT)", "--grpc-ip", "$(CILANTRO_SERVICE_SERVICE_HOST)", "--grpc-client-id", app_name],
                                                     container_image_pull_policy="Always")
        return [cilantroclient_dep]