"""
    A data source whose returned reward is a function of its resource allocation.
    -- romilbhardwaj
    -- kirthevasank
"""

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


AVG_LOAD = 13


class AllocationBasedDataSource(BaseDataSource):
    """
    Reads kubernetes state to get resource allocation and return metrics as a function
    of this allocation.
    """

    def __init__(self, app_name, namespace, load_file=None):
        """ Data source whose return values are a function of a resource allocation.
            :param app_name: Name of the (self) application to monitor in kubernetes.
            :param namespace: Namespace to use for kubernetes.
            :param load_file: Load file for the workload
        """
        super().__init__()
        self.app_name = app_name
        self.namespace = namespace
        # Load k8s config and create API objects
        load_k8s_config()
        self.appsapi = client.AppsV1Api()
        # workload set up
        logger.info('Load File received: "%s"', str(load_file))
        if not ((load_file is None) or (load_file == '')):
            with open(load_file, 'r') as data_read_file:
                lines = data_read_file.read().splitlines()
            self.loads = [float(elem) for elem in lines]
            self._load_file_cycle_idx = 0
            self._curr_load_cycle_vals = []
            self.len_loads = len(self.loads)
        else:
            self.loads = None
        self.last_get_time = time.time()

    def get_allocation(self):
        """ Obtain allocation. """
        deps = self.appsapi.list_namespaced_deployment(namespace=self.namespace)
        current_allocations = {d.metadata.name: d.status.replicas for d in deps.items}
        if self.app_name not in current_allocations:
            raise KeyError(f"App {self.app_name} not in current allocations. Currently running " \
                           f"apps: {list(current_allocations.keys())}")
        return current_allocations[self.app_name]

    def get_load(self):
        """ Obtain load. """
        if self.loads is None:
            curr_load = (AVG_LOAD - 0.1) + 0.2 * np.random.random()
        else:
            if len(self._curr_load_cycle_vals) == 0:
                self._load_file_cycle_idx += 1
                self._curr_load_cycle_vals = self.loads[:]
                if self._load_file_cycle_idx % 2 == 0:
                    self._curr_load_cycle_vals.reverse()
            curr_load = self._curr_load_cycle_vals.pop(0)
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
        if not curr_alloc:
            curr_alloc = 0
        curr_load = self.get_load()
        reward, sigma = self._get_reward_and_sigma_for_alloc_and_load(curr_alloc, curr_load)
        sigma_multiplier = 0.5 + np.random.random()
        sigma *= sigma_multiplier # using different sigma to test robustness to misspecification
        data = {
            'load': curr_load,
            'alloc': curr_alloc,
            'reward': reward,
            'sigma': sigma,
            'event_start_time': self.last_get_time,
            'event_end_time': now,
            'debug': "Hello world"
        }
        logger.info('At time %0.4f, returning data %s.', now, str(data))
        self.last_get_time = now
        return data

    def _get_reward_and_sigma_for_alloc_and_load(self, curr_alloc, curr_load):
        """ Returns reward and sigma for the allocation and load. """
        raise NotImplementedError('Implement in a child class.')

    @classmethod
    def add_args_to_parser(cls, parser: ArgumentParser):
        """ Adds arguments to the parser. """
        parser.add_argument('--k8s-app-name', '-an', type=str, default="",
                            help='Kubernetes app name to track for allocation changes.')
        parser.add_argument('--k8s-namespace', '-ns', type=str, default="default",
                            help='Kubernetes namespace to operate in.')
        parser.add_argument('--workload-type', '-wt', type=str,
                            help='Workload type for dummy workload.')
        parser.add_argument('--load-file', '-lf', type=str, default="",
                            help='File for the loads.')


class GLMAllocationBasedDataSource(AllocationBasedDataSource):
    """ A K8s dummy data source based on GLMs. """

    def __init__(self, app_name, namespace, mu, coeff, load_file=None):
        """ Constructor. """
        super().__init__(app_name, namespace, load_file)
        self.mu = mu
        self.coeff = coeff

    def _get_reward_and_sigma_for_alloc_and_load(self, curr_alloc, curr_load):
        """ Returns reward and sigma for the allocation and load. """
        if curr_load <= 1:
            int_load = 1
        else:
            int_load = int(curr_load)
        payoff = self.mu((curr_alloc/int_load) * self.coeff)
        successful_queries = np.random.random((int_load, )) < payoff
        sigma = 1/(2 * np.sqrt(int_load))
        reward = successful_queries.mean()
        return reward, sigma


class TanhAllocationBasedDataSource(GLMAllocationBasedDataSource):
    """ A K8s dummy data source based on the Tanh function. """

    def __init__(self, app_name, namespace, coeff, load_file=None):
        """ Constructor. """
        super().__init__(app_name, namespace, np.tanh, coeff, load_file)


class VPIAllocationBasedDataSource(GLMAllocationBasedDataSource):
    """ A K8s dummy data source based on the Tanh function. """

    def __init__(self, app_name, namespace, coeff, load_file=None, poly_factor=1):
        """ Constructor. """
        mu = lambda x: 1 - 1/(1 + x**poly_factor)
        super().__init__(app_name, namespace, mu, coeff, load_file)


class LogisticAllocationBasedDataSource(GLMAllocationBasedDataSource):
    """ A K8s dummy data source based on the Tanh function. """

    def __init__(self, app_name, namespace, coeff, bias, load_file=None):
        """ Constructor. """
        mu = lambda x: 1 / (1 + np.exp(-(x - bias)))
        super().__init__(app_name, namespace, mu, coeff, load_file)


class LinearAllocationBasedDataSource(AllocationBasedDataSource):
    """ Kubernetes based data source. """

    def __init__(self, app_name, namespace, coeff, sigma_max, load_file=None):
        """ Constructor. """
        super().__init__(app_name, namespace, load_file)
        self.coeff = coeff
        self.sigma_max = sigma_max

    def _get_reward_and_sigma_for_alloc_and_load(self, curr_alloc, curr_load):
        """ Returns reward and sigma for the allocation and load. """
        sigma = self.sigma_max * (0.5 + 0.5 * np.random.random())
        payoff = (curr_alloc/curr_load) * self.coeff
        reward = max(0, payoff + sigma * np.random.normal())
        return reward, sigma

