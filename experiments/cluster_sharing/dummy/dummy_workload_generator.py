"""
    Workload generator for proportional allocation
    -- romilbhardwaj
    -- kirthevasank
"""

from typing import List, Union
from kubernetes.client import V1Deployment, V1Service
# Local
from cilantro.workloads.base_workload_generator import BaseWorkloadGenerator
from cilantro.workloads.k8s_utils import get_template_deployment


class DummyWorkloadGenerator(BaseWorkloadGenerator):
    """
        A dummy workload generator.
        Its cilantro client return utility and other metrics as function of the k8s allocation.
    """
    # pylint: disable=arguments-differ

    def __init__(self, cluster_type=None, container_image=None):
        """ Constructor. """
        self._cluster_type = cluster_type
        if container_image:
            self.container_image = container_image
        elif cluster_type == 'eks':
            self._container_image = "public.ecr.aws/cilantro/cilantro:latest"
        elif cluster_type == 'kind':
            self._container_image = "public.ecr.aws/cilantro/cilantro:latest"
        else:
            raise ValueError('Invalid input for cluster_type(%s) and/or container_image(%s)'%(
                str(cluster_type), str(container_image)))
        super().__init__()

    def generate_workload_server_objects(self,
                                         app_name: str,
                                         threshold: float,
                                         app_weight: float,
                                         app_unit_demand,
                                         *args,
                                         **kwargs
                                         )-> List[Union[V1Deployment, V1Service]]:
        """ Generates workload server objects. """
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

    def generate_workload_client_objects(self, *args, **kwargs) -> List[
        Union[V1Deployment, V1Service]]:
        """ Generates workload client objects. """
        return []   # No workload clients for this workload since its a dummy

    def generate_cilantro_client_objects(self,
                                         app_name:str,
                                         workload_type:str,
                                         *args,
                                         **kwargs) -> List[Union[V1Deployment, V1Service]]:
        cilantroclient_dep = get_template_deployment(
            app_name=app_name + "--cc",
            is_workload="false",
            container_image=self._container_image,
            container_ports=[10000],
            container_command=["python",
                               "/cilantro/experiments/cluster_sharing/dummy/dummy_workload_driver.py"],
            container_args=["--k8s-app-name", app_name,
                            "--workload-type", workload_type,
                            "--load-file", "/cilantro/experiments/cluster_sharing/dummy/twitter_1476_data",
                            "--grpc-port", "$(CILANTRO_SERVICE_SERVICE_PORT)",
                            "--grpc-ip", "$(CILANTRO_SERVICE_SERVICE_HOST)",
                            "--grpc-client-id", app_name],
            container_image_pull_policy="Always"
            )
        return [cilantroclient_dep]

