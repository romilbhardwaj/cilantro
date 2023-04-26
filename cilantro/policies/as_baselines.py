"""
    Implements k8s's default autoscaling policy.
    -- kirthevasank
    -- romilbhardwaj
"""

import logging
import numpy as np
# Local
from cilantro.policies.autoscaling import AutoScalingBasePolicy

logger = logging.getLogger(__name__)


class K8sAutoScaler(AutoScalingBasePolicy):
    """ K8s autoscaling. """

    def __init__(self, env, resource_quantity, performance_recorder_bank,
                 num_recent_event_logs_for_util_computation=20,
                 grid_size_for_util_computation=5, scaling_coeff=1.0):
        """ Constructor. """
        super().__init__(env, resource_quantity, load_forecaster_bank=None)
        self.performance_recorder_bank = performance_recorder_bank
        self.num_recent_event_logs_for_util_computation = num_recent_event_logs_for_util_computation
        self.grid_size_for_util_computation = grid_size_for_util_computation
        self.scaling_coeff = scaling_coeff # Using 1 seems to cause wide fluctuations
        self._last_alloc = None

    def _get_autoscaling_resource_allocation_for_loads(self, load, *args, **kwargs):
        """ Get autoscaling resource allocation. """
        # pylint: disable=unused-argument
        # pylint: disable=broad-except
        if self.round_idx == 0:
            allocs = {key: 1 for key in self.env.leaf_nodes}
        else:
            try:
                recent_allocs_and_util_metrics = \
                    self.performance_recorder_bank.get_recent_allocs_and_util_metrics(
                        num_recent_event_logs=self.num_recent_event_logs_for_util_computation,
                        grid_size=self.grid_size_for_util_computation)
                if recent_allocs_and_util_metrics is None:
                    allocs = self._last_alloc
                else:
                    allocs = {}
                    for leaf_path, leaf in self.env.leaf_nodes.items():
                        curr_replicas = recent_allocs_and_util_metrics[-1]['alloc'][leaf_path]
                        curr_reward = recent_allocs_and_util_metrics[-1]['leaf_rewards'][leaf_path]
    #                     allocs[leaf_path] = int(curr_replicas * leaf.threshold / curr_reward)
#                         allocs[leaf_path] = int(np.round(self.scaling_coeff * curr_replicas *
#                                                          leaf.threshold / curr_reward))
                        allocs[leaf_path] = int(np.ceil(self.scaling_coeff * curr_replicas *
                                                        leaf.threshold / curr_reward))
            except Exception as e:
                # Using an except here since performance recorders fail early on.
                allocs = self._last_alloc
                logger.info('[%s] returning last alloc as exception occurred: %s', self, str(e))
        self._last_alloc = allocs
        return allocs


class PIDAutoScaler(AutoScalingBasePolicy):
    """ PID controller. """

    def __init__(self, env, resource_quantity, performance_recorder_bank,
                 p_coeff=10, i_coeff=0.000, d_coeff=4,
                 num_recent_event_logs_for_util_computation=20,
                 grid_size_for_util_computation=5):
        """ Controller. """
        super().__init__(env, resource_quantity, load_forecaster_bank=None)
        self.performance_recorder_bank = performance_recorder_bank
        self.num_recent_event_logs_for_util_computation = num_recent_event_logs_for_util_computation
        self.grid_size_for_util_computation = grid_size_for_util_computation
        self.p_coeff = p_coeff
        self.i_coeff = i_coeff
        self.d_coeff = d_coeff
        self._last_alloc = None
        self._curr_errs = {key: 0 for key in self.env.leaf_nodes}
        self._sum_errs = {key: 0 for key in self.env.leaf_nodes}
        self._diff_errs = {key: 0 for key in self.env.leaf_nodes}

    def _update_error_coeffs(self, recent_allocs_and_util_metrics):
        """ Update error parameters. """
        for leaf_path, leaf_node in self.env.leaf_nodes.items():
            curr_err = (leaf_node.threshold -
                        recent_allocs_and_util_metrics[-1]['leaf_rewards'][leaf_path])
            self._diff_errs[leaf_path] = curr_err - self._curr_errs[leaf_path]
            self._sum_errs[leaf_path] += curr_err
            # Swap current errors ----------------------------------------------------
            self._curr_errs[leaf_path] = curr_err

    def _get_autoscaling_resource_allocation_for_loads(self, load, *args, **kwargs):
        """ Get autoscaling resource allocation. """
        # pylint: disable=unused-argument
        # pylint: disable=broad-except
        if self.round_idx == 0:
            allocs = {key: 1 for key in self.env.leaf_nodes}
        else:
            try:
                recent_allocs_and_util_metrics = \
                    self.performance_recorder_bank.get_recent_allocs_and_util_metrics(
                        num_recent_event_logs=self.num_recent_event_logs_for_util_computation,
                        grid_size=self.grid_size_for_util_computation)
                if recent_allocs_and_util_metrics is None:
                    allocs = self._last_alloc
                else:
                    self._update_error_coeffs(recent_allocs_and_util_metrics)
                    allocs = {}
                    for leaf_path in self.env.leaf_nodes:
                        curr_replicas = recent_allocs_and_util_metrics[-1]['alloc'][leaf_path]
                        change = curr_replicas * (
                            self.p_coeff * self._curr_errs[leaf_path] +
                            self.i_coeff * self._sum_errs[leaf_path] +
                            self.d_coeff * self._diff_errs[leaf_path])
                        allocs[leaf_path] = int(max(1, curr_replicas + change))
            except Exception as e:
                # Using an except here since performance recorders fail early on.
                allocs = self._last_alloc
                logger.info('[%s] returning last alloc as exception occurred: %s', self, str(e))
        self._last_alloc = allocs
        return allocs


class DS2AutoScaler(AutoScalingBasePolicy):
    """ DS2 auto scaler. """

    def __init__(self, env, resource_quantity, load_forecaster_bank, performance_recorder_bank,
                 num_recent_event_logs_for_util_computation=20,
                 grid_size_for_util_computation=5, scaling_coeff=1.0):
        """ Constructor. """
        super().__init__(env, resource_quantity, load_forecaster_bank)
        self.performance_recorder_bank = performance_recorder_bank
        self.num_recent_event_logs_for_util_computation = num_recent_event_logs_for_util_computation
        self.grid_size_for_util_computation = grid_size_for_util_computation
        self.scaling_coeff = scaling_coeff # Using 1 seems to cause wide fluctuations
        self._last_alloc = None

    def _get_autoscaling_resource_allocation_for_loads(self, load, *args, **kwargs):
        """ Get autoscaling resource allocation. """
        # pylint: disable=unused-argument
        # pylint: disable=broad-except
        if self.round_idx == 0:
            allocs = {key: 1 for key in self.env.leaf_nodes}
        else:
            try:
                recent_allocs_and_util_metrics = \
                    self.performance_recorder_bank.get_recent_allocs_and_util_metrics(
                        num_recent_event_logs=self.num_recent_event_logs_for_util_computation,
                        grid_size=self.grid_size_for_util_computation)
                if recent_allocs_and_util_metrics is None:
                    allocs = self._last_alloc
                else:
                    allocs = {}
                    for leaf_path in self.env.leaf_nodes:
                        curr_replicas = recent_allocs_and_util_metrics[-1]['alloc'][leaf_path]
                        # Estimate the processing rate ---------------------------------------
                        est_processing_rate = \
                            (recent_allocs_and_util_metrics[-1]['loads'][leaf_path] *
                             recent_allocs_and_util_metrics[-1]['leaf_rewards'][leaf_path])
                        allocs[leaf_path] = int(np.round(
                            self.scaling_coeff * load[leaf_path] * curr_replicas /
                            est_processing_rate))
            except Exception as e:
                # Using an except here since performance recorders fail early on.
                allocs = self._last_alloc
                logger.info('[%s] returning last alloc as exception occurred: %s', self, str(e))
#                 raise ValueError
        self._last_alloc = allocs
        return allocs
