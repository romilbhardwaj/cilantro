from typing import List, Union

from kubernetes.client import V1Service, V1Deployment


class BaseWorkloadGenerator(object):
    """
    This class defines the base workload generator for kubernetes workloads.
    Each cilantro workload has three components:
    * Workload servers, which are scaled by cilantro
    * Workload client, which generates the queries (load) for the workload servers
    * Cilantro client, which reads the workload client and/or cluster state to compute
      utility metrics and report them to cilantro scheduler.

    WorkloadGenerators implement methods to return kubernetes objects for each of
    these components.
    """
    def __init__(self):
        pass

    def generate_workload_server_objects(self, *args, **kwargs) -> List[Union[V1Deployment, V1Service]]:
        raise NotImplementedError("Implement in a child class.")

    def generate_workload_client_objects(self, *args, **kwargs) -> List[Union[V1Deployment, V1Service]]:
        raise NotImplementedError("Implement in a child class.")

    def generate_cilantro_client_objects(self, *args, **kwargs) -> List[Union[V1Deployment, V1Service]]:
        raise NotImplementedError("Implement in a child class.")