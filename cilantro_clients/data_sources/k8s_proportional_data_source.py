"""
A data source whose returned reward is a function of its resource allocation.
"""
import random
import time
from argparse import ArgumentParser
from typing import Dict
from kubernetes import client, config, watch
import logging

from cilantro.backends.k8s.kubernetes_manager import load_k8s_config
from cilantro_clients.data_sources.base_data_source import BaseDataSource

logger = logging.getLogger(__name__)
client.rest.logger.setLevel(logging.WARNING)

class AllocationBasedDataSource(BaseDataSource):
    def __init__(self,
                 load_fn: callable = None,
                 alloc_fn: callable = None,
                 reward_fn: callable = None,
                 sigma_fn: callable = None):
        """
        Data source whose return values are a function of a resource allocation.
        :param load_fn: callable used to compute load. Has allocation parameter. Eg. Random fn.
        :param alloc_fn: callable used to compute allocation. Has allocation parameter. Eg. identity fn.
        :param reward_fn: callable used to compute reward. Has allocation parameter. Eg. Linear proportional fn.
        :param sigma_fn: callable used to compute sigma. Has allocation parameter. Eg. Constant fn.
        """
        self.load_fn = load_fn if load_fn else lambda alloc: random.randint(0, 10)
        self.alloc_fn = alloc_fn if alloc_fn else lambda alloc: alloc
        self.reward_fn = reward_fn if reward_fn else lambda alloc: alloc
        self.sigma_fn = sigma_fn if sigma_fn else lambda alloc: 1
        self.last_get_time = 0

    def get_allocation(self):
        raise NotImplementedError('Implement in child class')

    def get_data(self) -> Dict:
        now = time.time()
        try:
            alloc = self.get_allocation()
        except KeyError as e:
            logger.error(f"KeyError occurred in get_allocation. Returning none and not publishing data. Error: {str(e)}")
            return None
        data = {
            'load': self.load_fn(alloc),
            'alloc': self.alloc_fn(alloc),
            'reward': self.reward_fn(alloc),
            'sigma': self.sigma_fn(alloc),
            'event_start_time': self.last_get_time,
            'event_end_time': now
        }
        self.last_get_time = now
        return data
    
class K8sProportionalDataSource(AllocationBasedDataSource):
    def __init__(self,
                 app_name: str,
                 *args,
                 namespace: str = 'default',
                 **kwargs):
        """
        Reads kubernetes state to get resource allocation and return metrics as a function
        of this allocation.
        :param app_name: Name of the (self) application to monitor in kubernetes.
        :param args: Args to AllocationBasedDataSource
        :param namespace: Namespace to use for kubernetes.
        :param kwargs: kwargs to AllocationBasedDataSource
        """
        self.app_name = app_name
        self.namespace = namespace

        # Load k8s config and create API objects
        load_k8s_config()
        self.appsapi = client.AppsV1Api()

        super(K8sProportionalDataSource, self).__init__(*args, **kwargs)

    def get_allocation(self):
        deps = self.appsapi.list_namespaced_deployment(namespace=self.namespace)
        current_allocations = {d.metadata.name: d.status.replicas for d in deps.items}
        if self.app_name not in current_allocations:
            raise KeyError(f"App {self.app_name} not in current allocations. Currently running apps: {list(current_allocations.keys())}")
        return current_allocations[self.app_name]

    @classmethod
    def add_args_to_parser(self,
                           parser: ArgumentParser):
        parser.add_argument('--k8s-app-name', '-an', type=str, default="",
                            help='Kubernetes app name to track for allocation changes.')
        parser.add_argument('--k8s-namespace', '-ns', type=str, default="default",
                            help='Kubernetes namespace to operate in.')
