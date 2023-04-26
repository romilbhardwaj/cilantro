"""
    Policy class.
    -- kirthevasank
    -- romilbhardwaj
"""


# from argparse import Namespace
import logging
from typing import Dict

import numpy as np


logger = logging.getLogger(__name__)


class BasePolicy:
    """ A policy class for online learning in resource allocation. """

    def __init__(self, env, resource_quantity, load_forecaster_bank=None, alloc_granularity=None):
        """ Constructor.
            env: The hierarchical environment used for allocation.
            resource_quantity: The quantity of the resource to be allocated.
            alloc_granularity: The granularity of each allocation. Any allocation should be an
                               integral amount of this value.
        """
        self.env = env
        self.resource_quantity = resource_quantity
        self.load_forecaster_bank = load_forecaster_bank
        self.alloc_granularity = alloc_granularity
        self.num_alloc_quanta = int(np.floor(resource_quantity / alloc_granularity)) \
                                if alloc_granularity else None
        self.round_idx = None

    def initialise(self):
        """ Initialise. """
        self.round_idx = 0
        self._policy_initialise()

    def _policy_initialise(self):
        """ Initialisation in a child class. """
        raise NotImplementedError('Implement in a child class.')

    def get_resource_allocation(self, curr_loads=None, *args, **kwargs) -> Dict[str, float]:
        """ Get allocations for the given loads.
            If curr_loads is None, will use the loads in the environment.
            Return allocations are a dictionary of app_path: float allocation."""
#         logger.info(f'Given current loads {curr_loads}')
        if not curr_loads:
            if self.load_forecaster_bank:
                curr_loads = {}
                for key, load_forecaster in self.load_forecaster_bank.enumerate():
                    if load_forecaster is None:
                        # App exists in the env but hasn't been initialized yet.
                        # Can happen when the app is launched
                        # after the env is defined.
                        load_ucb = None
                    else:
                        _, _, load_ucb = load_forecaster.forecast()
                    curr_loads[key] = load_ucb
            else:
                curr_loads = {leaf_path: None for leaf_path in self.env.leaf_nodes}
#         logger.info(f'Final current loads {curr_loads}')
        if len(curr_loads) != 0:
            # There are jobs in the system
            ret = self._get_resource_allocation_for_loads(curr_loads, *args, **kwargs)
        else:
            # No jobs in the system
            ret = {}
        self.round_idx += 1
        return ret

    def _get_resource_allocation_for_loads(self, loads, *args, **kwargs):
        """ Returns the allocations for the loads. """
        raise NotImplementedError('Implement in a child class.')

    def _get_final_allocations_from_ratios(self, alloc_ratios, min_alloc_quanta=0):
        """ Returns final allocations from ratios depending on whether the allocations should be
            discrete or not.
        """
        if self.num_alloc_quanta:
            ret_quanta = {key: val * self.num_alloc_quanta for key, val in alloc_ratios.items()}
            ret_disc_quanta = {key: max(min_alloc_quanta, int(np.floor(val)))
                               for key, val in ret_quanta.items()}
            ret_remainders = {key: max(0, val - ret_disc_quanta[key])
                              for key, val in ret_quanta.items()}
            num_remaining_quanta = self.num_alloc_quanta - \
                                   sum([val for _, val in ret_disc_quanta.items()])
            if num_remaining_quanta > 0:
                chosen_leafs = self._randomised_rounding(ret_remainders, num_remaining_quanta)
                for key in chosen_leafs:
                    ret_disc_quanta[key] += 1
            # Finally renormalise to fit the total resource amount
            ret = {key: val * self.alloc_granularity for key, val in ret_disc_quanta.items()}
        else:
            ret = {leaf: alloc_ratios[leaf] * self.resource_quantity for leaf in alloc_ratios}
        return ret

    @classmethod
    def _randomised_rounding(cls, unnormalised_prob_dict, num_quanta):
        """ A utility for randomised rounding. Can be used by any inherting policy which computes
            fractional allocations but needs to allocate resources in discrete quanta.
        """
        keys = list(unnormalised_prob_dict.keys()) # make a list of keys, fix its order
        unnormalised_probs = np.array([unnormalised_prob_dict[key] for key in keys])
        prob_sum = unnormalised_probs.sum()
        if prob_sum <= 0:
            normalised_probs = None
        else:
            normalised_probs = unnormalised_probs / unnormalised_probs.sum()
        try:
            chosen_keys = np.random.choice(keys, size=num_quanta, replace=False, p=normalised_probs)
        except ValueError:
            chosen_keys = np.random.choice(keys, size=num_quanta, replace=True, p=normalised_probs)
        return chosen_keys


def get_str_true_unit_demands_and_CIs(leaf_nodes, policy_leaf_unit_demand_bounds):
    """ Prints out the leaf_node demands and bounds learned by the policy. """
    ret_list = []
    for ag_idx, (leaf_node, bounds) in enumerate(zip(leaf_nodes,
                                                     policy_leaf_unit_demand_bounds)):
        if leaf_node.unit_demand < bounds.lb:
            succ_str = '<'
        elif leaf_node.unit_demand > bounds.ub:
            succ_str = '>'
        else:
            succ_str = '1'
        ret_list.append('   Ag-%d: %0.5f, (%0.5f, %0.5f), gap=%0.5e, %s'%(
            ag_idx, leaf_node.unit_demand, bounds.lb, bounds.ub, bounds.ub-bounds.lb, succ_str))
    return '\n'.join(ret_list)

