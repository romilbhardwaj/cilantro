"""
    A data source whose returned reward is a function of its resource allocation.
    -- romilbhardwaj
    -- kirthevasank
"""

# pylint: disable=import-error

from argparse import ArgumentParser
import logging
import time
from typing import Dict
from kubernetes import client
import numpy as np
# Local
from cilantro.backends.k8s.kubernetes_manager import load_k8s_config
from cilantro_clients.data_sources.base_data_source import BaseDataSource

logger = logging.getLogger(__name__)
client.rest.logger.setLevel(logging.WARNING)


UD_THRESH_RATIO = 1/3
AVG_LOAD = 4


def get_abs_unit_demand(threshold):
    """ compute the unit demand from the threshold. """
    return threshold * UD_THRESH_RATIO


class AllocationBasedDataSource(BaseDataSource):
    """ An allocation based data source. """

    def __init__(self, load_file=None, sigma_max=0.3):
        """ Data source whose return values are a function of a resource allocation.
        """
        self.ud_thresh_ratio = UD_THRESH_RATIO
        logger.info('Load file received: "%s"', str(load_file))
        if not ((load_file is None) or (load_file == '')):
            with open(load_file, 'r') as data_read_file:
                lines = data_read_file.read().splitlines()
            self.loads = [float(elem) for elem in lines]
            self.load_idx = 0
            self.len_loads = len(self.loads)
        else:
            self.loads = None
        self.last_get_time = 0
        self.sigma_max = sigma_max
        super().__init__()

    def get_allocation(self):
        """ Obtain allocation. """
        raise NotImplementedError('Implement in child class')

    def get_load(self):
        """ Obtain load. """
        if self.loads is None:
            curr_load = (AVG_LOAD - 0.1) + 0.2 * np.random.random()
        else:
            self.load_idx += 1
            curr_load = self.loads[(self.load_idx - 1) % self.len_loads]
        return curr_load

    def get_data(self) -> Dict:
        """ Obtain data. """
        now = time.time()
        try:
            curr_alloc = self.get_allocation()
        except KeyError as e:
            log_msg = f"KeyError occurred in get_allocation or get_load. Returning none and " \
                      f"not publishing data. Error: {str(e)}"
            logger.error(log_msg)
            return None
        curr_load = self.get_load()
        sigma = self.sigma_max * (0.5 + 0.5 * np.random.random())
        payoff = (curr_alloc/curr_load) / self.ud_thresh_ratio
        reward = payoff + sigma * np.random.normal()
        data = {
            'load': curr_load,
            'alloc': curr_alloc,
            'reward': reward,
            'sigma': 2 * sigma, # using 2*sigma to test how robust it is to misspecification
            'event_start_time': self.last_get_time,
            'event_end_time': now
        }
        logger.info('Returning data %s.', str(data))
        self.last_get_time = now
        return data


class K8sProportionalDataSource(AllocationBasedDataSource):
    """ Kubernetes based data source. """

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
        super().__init__(*args, **kwargs)

    def get_allocation(self):
        """ Obtain allocation. """
        deps = self.appsapi.list_namespaced_deployment(namespace=self.namespace)
        current_allocations = {d.metadata.name: d.status.replicas for d in deps.items}
        if self.app_name not in current_allocations:
            raise KeyError(f"App {self.app_name} not in current allocations. Currently running " \
                           f"apps: {list(current_allocations.keys())}")
        return current_allocations[self.app_name]

    @classmethod
    def add_args_to_parser(cls, parser: ArgumentParser):
        """ Adds arguments to the parser. """
        parser.add_argument('--k8s-app-name', '-an', type=str, default="",
                            help='Kubernetes app name to track for allocation changes.')
        parser.add_argument('--k8s-namespace', '-ns', type=str, default="default",
                            help='Kubernetes namespace to operate in.')
        parser.add_argument('--load-file', '-lf', type=str, default="",
                            help='File for the loads.')

