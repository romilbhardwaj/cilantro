"""
    A policy which simply alocates proportional to entitlements.
    -- kirthevasank
    -- romilbhardwaj
"""

import logging
import numpy as np
# Local
from cilantro.policies.base_policy import BasePolicy
from cilantro.policies.mmf import HMMF


logger = logging.getLogger(__name__)


class PropFairness(BasePolicy):
    """ Allocates in proportion to the entitlements in an hierarchical environment. """

    def _policy_initialise(self):
        """ Initialisation. """
        # pylint: disable=import-outside-toplevel
        self.hmmf_policy = HMMF(self.env, resource_quantity=self.resource_quantity,
                                load_forecaster_bank=self.load_forecaster_bank,
                                alloc_granularity=None)
        self.hmmf_policy.initialise()

    def _get_resource_allocation_for_loads(self, loads):
        """ Computes allocations for loads. """
        # pylint: disable=arguments-differ
        stored_loads = {}
        stored_unit_demands = {}
        loads_to_pass_to_hmmf = {}
        for leaf_path in self.env.leaf_nodes:
            leaf = self.env.leaf_nodes[leaf_path]
            stored_loads[leaf_path] = loads.get(leaf_path, leaf.get_curr_load())
            stored_unit_demands[leaf_path] = leaf.unit_demand
            leaf.set_curr_load(np.inf)
            leaf.set_unit_demand(np.inf)
            loads_to_pass_to_hmmf[leaf_path] = np.inf
        allocs_pre = self.hmmf_policy.get_resource_allocation(curr_loads=loads_to_pass_to_hmmf)
        allocs = {leaf_path:val/self.resource_quantity for leaf_path, val in allocs_pre.items()}
        for (leaf_path, ld) in stored_loads.items(): # Copy old values back over
            leaf = self.env.leaf_nodes[leaf_path]
            leaf.set_curr_load(ld)
            leaf.set_unit_demand(stored_unit_demands[leaf_path])
        # Now compute the unnormalised version
        final_allocs = self._get_final_allocations_from_ratios(allocs)
        return final_allocs

