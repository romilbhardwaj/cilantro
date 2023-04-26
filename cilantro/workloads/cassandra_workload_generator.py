from typing import List, Union

from kubernetes.client import V1Deployment, V1Service, V1ServicePort

from cilantro.workloads.base_workload_generator import BaseWorkloadGenerator
from cilantro.workloads.cassandra_k8s_utils import get_cassandra_seed_template_deployment, \
    get_cassandra_seed_template_service, get_cassandra_server_template_deployment, \
    get_cassandra_server_template_service, get_cassandra_client_template_deployment
from cilantro.workloads.k8s_utils import get_template_deployment, get_template_service


class CassandraWorkloadGenerator(BaseWorkloadGenerator):
    """
    This is a workload generator for the data-serving workload from cloudsuite benchmarks.
    See implementation at https://github.com/romilbhardwaj/cloudsuite/tree/k8s
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
                                         threshold: float,
                                         app_weight: float,
                                         app_unit_demand: float,
                                         *args,
                                         **kwargs
                                         )-> List[Union[V1Deployment, V1Service]]:
        # Get seeds - this is a deployment required to bootstrap the workload.
        seed_deployment = get_cassandra_seed_template_deployment(app_name)
        seed_svc = get_cassandra_seed_template_service(app_name)

        # Get server deployment - this is the actual workload.
        server_deployment = get_cassandra_server_template_deployment(app_name,
                                                                     seed_svc_name=seed_svc.metadata.name,
                                                                     threshold=str(threshold),
                                                                     app_weight=str(app_weight),
                                                                     app_unit_demand=str(app_unit_demand)
                                                                     )
        server_service = get_cassandra_server_template_service(app_name)

        return [seed_svc, seed_deployment, server_service, server_deployment]

    def generate_workload_client_objects(self,
                                         app_name: str,
                                         threshold: float,
                                         *args,
                                         server_svc: str = None,
                                         **kwargs) -> List[Union[V1Deployment, V1Service]]:
        """
        Generates workload client objects. The cilantro client is embedded in the same pod.
        :param app_name: Name of the app in the heirarchy.
        :param args: other args to pass to get_cassandra_client_template_deployment
        :param server_svc: Service name of the cassandra servers. If not specified, infers it from app_name
        :param kwargs: other kwargs to pass to get_cassandra_client_template_deployment
        :return:
        """
        if server_svc is None:
            # If not specified, infer from app_name
            server_svc = app_name + "-svc"

        client_dep = get_cassandra_client_template_deployment(app_name,
                                                              server_svc,
                                                              threshold
                                                              *args,
                                                              **kwargs)
        return [client_dep]

    def generate_cilantro_client_objects(self,
                                         app_name:str,
                                         *args,
                                         **kwargs) -> List[Union[V1Deployment, V1Service]]:
        return [] # No cilantro clients for this workload since it is co-located with the workload client