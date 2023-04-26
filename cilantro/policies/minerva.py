"""
    Minerva policy for resource allocation.
    -- kirthevasank
    -- romilbhardwaj
"""

import logging
from copy import deepcopy
from cilantro.policies.base_policy import BasePolicy
from cilantro.policies.mmf import max_min_fairness

logger = logging.getLogger(__name__)


class Minerva(BasePolicy):
    """ Minerva. """

    def __init__(self, env, resource_quantity, load_forecaster_bank, reward_forecaster_bank,
                 *args, **kwargs):
        """ Constructor. """
        super().__init__(env, resource_quantity, load_forecaster_bank, *args, **kwargs)
        # Define variables
        self.reward_forecaster_bank = reward_forecaster_bank
        self.last_norm_rewards = None
        self.last_utils = None
        self.last_allocs = None

    def _policy_initialise(self):
        """ Initialise child policy. """
        pass

    def _get_resource_allocation_for_loads(self, loads, *args, **kwargs):
        """ Returns the allocations for the loads. """
        capacities = self.env.get_curr_capacities_for_all_leaf_nodes()
        cap_ratios = {leaf: capacities[leaf]/self.resource_quantity for leaf in capacities}
        if not self.last_allocs:
            alloc_ratios_without_caps = self.env.get_entitlements()
        else:
            curr_weights = {}
            total_sum = 0
            leaf_entitlements = self.env.get_entitlements()
            for leaf_path in self.env.leaf_nodes:
                reward_forecaster = self.reward_forecaster_bank.get(leaf_path)
                if reward_forecaster is None:
                    reward_est = 1
                else:
                    _, reward_est, _ = reward_forecaster.forecast()
                curr_leaf_thresh = self.env.leaf_nodes[leaf_path].threshold
                util_est = min(reward_est, curr_leaf_thresh) / curr_leaf_thresh
#                 util_est = reward_est / curr_leaf_thresh
                curr_weights[leaf_path] = (self.last_allocs[leaf_path] + 0.0001) / \
                                          (max(0, util_est) + 0.0001)
#                 curr_weights[leaf] = (self.last_allocs[leaf] + 0.001) / \
#                                      (self.last_rewards[leaf] + 0.001)
                curr_weights[leaf_path] *= leaf_entitlements[leaf_path]
                total_sum += curr_weights[leaf_path]
            alloc_ratios_without_caps = \
                {leaf_path: curr_weights[leaf_path]/total_sum for leaf_path in self.env.leaf_nodes}
        # TODO: this is a dirty fix for a 1000 node cluster ---------------------------------------
        # Make sure they all receive at least 1 resource
        min_thresh_for_each_leaf = 0.00101
        rem_res_for_all = 1 - min_thresh_for_each_leaf * len(self.env.leaf_nodes)
        for key in alloc_ratios_without_caps:
            alloc_ratios_without_caps[key] = min_thresh_for_each_leaf + \
                                             rem_res_for_all * alloc_ratios_without_caps[key]
        # Call max_min_fairness with demand set to cap_ratios
        leaf_order = sorted(list(self.env.leaf_nodes))
        cap_dems = [cap_ratios[leaf] for leaf in leaf_order]
        entitlements = [alloc_ratios_without_caps[leaf] for leaf in leaf_order]
        logger.info('Using entitlements %0.4f, %s', sum(entitlements), entitlements)
        allocs_order = max_min_fairness(entitlements, cap_dems, total_resource=1.0)
        alloc_ratios = dict(zip(leaf_order, allocs_order))
        # Obtain finall allocations
        final_allocs = self._get_final_allocations_from_ratios(alloc_ratios, min_alloc_quanta=1)
        self.last_allocs = deepcopy(final_allocs)
        return final_allocs

