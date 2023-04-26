import logging
from typing import List, Union

import kubernetes
from kubernetes.client import V1Deployment, V1Service

from cilantro.backends.k8s.kubernetes_manager import load_k8s_config
from cilantro.core.henv import TreeEnvironment

logger = logging.getLogger(__name__)

class K8sWorkloadDeployer(object):
    """
    This class parses an environment with pre-populated leaf nodes to generate
    (and optionally execute) a kubernetes workload plan.
    This method requires that the leaf node stores a keys in the workload_info called "k8s_objects".
    This is a list of V1Deployment and/or V1Service that are required for the application to function.

    Typically, workload_info["k8s_objects"] would contain 4 objects: a workload server deployment, a workload server service,
    a workload client deployment and a cilantro client deployment.
    """

    def __init__(self):
        load_k8s_config()
        self.coreapi = kubernetes.client.CoreV1Api()
        self.appsapi = kubernetes.client.AppsV1Api()
        super(K8sWorkloadDeployer, self).__init__()

    def parse_env_to_workloads(self,
                               env: TreeEnvironment) -> List[Union[V1Deployment, V1Service]]:
        """
        Parses an environment's leaf nodes to create a list of all kubernetes objects
        to be created for this environment. Requires leaf.workload_info to contain
        "k8s_objects" key.
        :param env: Tree environment
        :return: List of V1Deployment and V1Service objects
        """
        all_k8s_objects = []
        for leaf_path, leaf in env.leaf_nodes.items():
            if "k8s_objects" in leaf.workload_info:
                all_k8s_objects.extend(leaf.get_workload_info("k8s_objects"))
            # For each leaf, we need to generate a workload deployment and cilantroclient deployment.
        logger.debug(f"Parsed {len(all_k8s_objects)} from {len(env.leaf_nodes)} leafs.")
        return all_k8s_objects

    def deploy_k8s_objects(self,
                        k8s_objects: List[Union[V1Deployment, V1Service]]):
        """
        Given a list of k8s_objects, deploys them onto the running k8s cluster.
        :param k8s_objects: List of V1Deployment or V1Service
        :return:
        """
        for obj in k8s_objects:
            if isinstance(obj, V1Deployment):
                self.appsapi.create_namespaced_deployment(namespace="default",
                                                          body=obj)
            elif isinstance(obj, V1Service):
                self.coreapi.create_namespaced_service(namespace="default",
                                                       body=obj)
            else:
                raise NotImplementedError(f"K8sSe object type {type(obj)} not supported.")
        logger.info(f"Deployed {len(k8s_objects)} kubernetes objects. Check dashboard for status.")

    def deploy_environment(self,
                           env: TreeEnvironment):
        """
        Parses a TreeEnvironment and deploys them in kubernetes.
        :param env:
        :return:
        """
        k8s_objects = self.parse_env_to_workloads(env)
        logger.debug(f"Parsed {len(k8s_objects)} to deploy. Attempting to deploy now..")
        self.deploy_k8s_objects(k8s_objects)
