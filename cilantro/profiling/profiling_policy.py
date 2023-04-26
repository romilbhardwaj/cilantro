"""
    Profiling policy, to profile jobs in an environment.
    -- kirthevasank
    -- romilbhardwaj
"""

import logging
import numpy as np
from cilantro.policies.base_policy import BasePolicy

logger = logging.getLogger(__name__)


class ProfilingPolicy(BasePolicy):
    """ Allocates in proportion to the entitlements in an hierarchical environment. """

    def __init__(self, env, resource_quantity, leaf_path_to_profile, load_forecaster_bank=None,
                 profiler_resource_allocations=None, alloc_granularity=None,
                 num_rounds_to_profile=np.inf, profile_grid_size=40):
        """ Constructor.
            num_rounds_to_profile: Number of rounds to profile for each allocation.
        """
        super().__init__(env, resource_quantity, load_forecaster_bank, alloc_granularity)
        self.round_idx = None
        if profiler_resource_allocations is None:
            profiler_resource_allocations = self._get_default_profiler_resource_allocations(
                resource_quantity, profile_grid_size)
        self.leaf_path_to_profile = leaf_path_to_profile
        self.profiler_resource_allocations = profiler_resource_allocations
        self.num_rounds_to_profile = num_rounds_to_profile
        self.list_of_alloc_vals_for_profiling = []
        self._number_of_cycles_profiled = 0

    @classmethod
    def _get_default_profiler_resource_allocations(cls, resource_quantity, profile_grid_size):
        """ Returns the default profiler resource allocation. """
        ret_log = [int(x) for x in np.logspace(0, np.log10(min(100, resource_quantity)),
                                               profile_grid_size)]
        ret_lin = [int(np.round(x)) for x in np.linspace(0, min(150, resource_quantity),
                                                         profile_grid_size)]
        ret = np.unique(ret_log + ret_lin)
        ret = [int(x) for x in ret]
        ret.sort()
#         ret = [1, 5, 10, 20, 30, 50, 75, 96]
        return ret

    def _policy_initialise(self):
        """ Initialise policy. """
        pass

    def _get_resource_allocation_for_loads(self, loads, *args, **kwargs):
        """ Returns the allocations for the loads. """
        # pylint: disable=logging-not-lazy
        allocs = {leaf_path: 0 for leaf_path in loads}
        if not self.list_of_alloc_vals_for_profiling:
            profiler_res_allocs = self.profiler_resource_allocations[:]
            np.random.shuffle(profiler_res_allocs)
            self.list_of_alloc_vals_for_profiling = profiler_res_allocs
            # Print out some info ---------------------------------------------------------------
            workload_type = self.env.leaf_nodes[self.leaf_path_to_profile].get_workload_info(
                'workload_type')
            logger.info(('Completed %d profiling rounds for leaf %s with workload_type %s. ' +
                         'Starting new profiling round with allocs %s.'),
                        self._number_of_cycles_profiled, self.leaf_path_to_profile, workload_type,
                        self.list_of_alloc_vals_for_profiling)
            self._number_of_cycles_profiled += 1
        leaf_alloc = self.list_of_alloc_vals_for_profiling.pop(0)
        allocs[self.leaf_path_to_profile] = leaf_alloc
        # Add remaining nodes to the other nodes randomly
        other_jobs = [x for x in loads if x != self.leaf_path_to_profile]
        if other_jobs:
            for _ in range(int(self.resource_quantity - leaf_alloc)):
                curr_job_sel = np.random.choice(other_jobs)
                allocs[curr_job_sel] += 1
        log_info_str = f"[ProfilingPolicy] Round {self.round_idx}, allocating {leaf_alloc} to " \
                       f"{self.leaf_path_to_profile}"
        logger.info(log_info_str)
        return allocs

