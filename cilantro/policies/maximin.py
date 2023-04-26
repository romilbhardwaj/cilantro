"""
    Utilities for maximin optimisation.
    -- kirthevasank
"""

import logging
import numpy as np
# Local
from cilantro.policies.base_policy import BasePolicy
from cilantro.policies.prop_fairness import PropFairness

logger = logging.getLogger(__name__)


def get_next_alloc_to_try_for_maximin(curr_alloc, curr_utils, resource_quantity,
                                      max_num_leafs_to_change=np.inf):
#                                       num_leafs_to_change_in_curr_step=-1):
    """ Returns the next allocation to try. """
    all_keys = list(curr_alloc)
    all_utils = [curr_utils[key] for key in all_keys]
    sorted_indices = np.argsort(all_utils)
    num_leafs = len(sorted_indices)
    sorted_keys = [all_keys[idx] for idx in sorted_indices]
    num_leafs_to_add_to = min(max_num_leafs_to_change, num_leafs//2)
    keys_to_add_to = [elem for elem in sorted_keys[:num_leafs_to_add_to] if
                      curr_alloc[elem] <= resource_quantity - 2]
    num_leafs_to_take_from = len(keys_to_add_to)
    keys_to_take_from = [elem for elem in sorted_keys[-num_leafs_to_take_from:] if
                         curr_alloc[elem] >= 2]
    num_leafs_to_be_changed = len(keys_to_take_from)
    keys_to_add_to = keys_to_add_to[:num_leafs_to_be_changed]
#     if num_leafs_to_change_in_curr_step > 0:
#         num_leafs_to_change_in_curr_step = min(num_leafs_to_change_in_curr_step,
#                                                num_leafs_to_be_changed)
#         keys_to_add_to = list(np.random.choice(keys_to_add_to,
#                                                num_leafs_to_change_in_curr_step,
#                                                replace=False))
#         keys_to_take_from = list(np.random.choice(keys_to_take_from,
#                                                   num_leafs_to_change_in_curr_step,
#                                                   replace=False))
    ret = dict(curr_alloc)
    for key in keys_to_take_from:
        ret[key] -= 1
    for key in keys_to_add_to:
        ret[key] += 1
    return ret


def optimise_func_maximin(func, env, resource_quantity, max_num_iters,
                          max_tries_before_reducing_size=5, return_all_allocs=False):
    """ Optimises a function with maximin.
        func takes a dictionary of allocations and returns a dictionary of utilitities.
    """
    leaf_nodes = list(env.leaf_nodes)
    num_leafs = len(leaf_nodes)
    # Obtain the initial allocation via the  -----------------------------------
    prop_fair_policy = PropFairness(env, resource_quantity, alloc_granularity=1)
    prop_fair_policy.initialise()
    curr_alloc = prop_fair_policy.get_resource_allocation()
    curr_alloc_list = [val for _, val in curr_alloc.items()]
    assert sum(curr_alloc_list) == resource_quantity
    # Some book-keeping before we commence the main loop -----------------------
    curr_max_num_leafs_to_change = max(1, num_leafs // 2)
    curr_utils = func(curr_alloc)
    num_equal_allocs = 0
    previous_evals = [None, curr_alloc]
    curr_best_min_util = -np.inf
    curr_best_alloc = None
    all_allocs = [curr_alloc]
    for _ in range(max_num_iters):
        next_alloc = get_next_alloc_to_try_for_maximin(
            curr_alloc, curr_utils, resource_quantity,
            max_num_leafs_to_change=curr_max_num_leafs_to_change)
        next_utils = func(next_alloc)
        to_compare_with = previous_evals.pop(0)
        num_equal_allocs += int(to_compare_with == next_alloc)
        # Update the optimum --------------------------------
        next_min_util = min([val for _, val in next_utils.items()])
        if next_min_util >= curr_best_min_util:
            curr_best_alloc = next_alloc
            curr_best_min_util = next_min_util
        # Test for convergence ------------------------------
        if num_equal_allocs >= max_tries_before_reducing_size:
            if curr_max_num_leafs_to_change == 1:
                break
            curr_max_num_leafs_to_change = max(1, curr_max_num_leafs_to_change // 2)
            num_equal_allocs = 0
        # Remaining book-keeping ----------------------------
        previous_evals.append(next_alloc)
        curr_alloc = next_alloc
        curr_utils = next_utils
        all_allocs.append(curr_alloc)
    if return_all_allocs:
        return all_allocs
    else:
        return curr_best_alloc


class EgalWelfareGreedy(BasePolicy):
    """ A greedy policy for maximising the egalitarian welfare. """

    def __init__(self, env, resource_quantity, performance_recorder_bank, load_forecaster_bank,
                 alloc_granularity=None, num_recent_event_logs_for_util_computation=20,
                 grid_size_for_util_computation=5, max_num_leafs_to_change_per_iter=9):
        """ Constructor. """
        super().__init__(env, resource_quantity, load_forecaster_bank, alloc_granularity)
        self.performance_recorder_bank = performance_recorder_bank
        self.num_recent_event_logs_for_util_computation = num_recent_event_logs_for_util_computation
        self.grid_size_for_util_computation = grid_size_for_util_computation
        self.max_num_leafs_to_change_per_iter = max_num_leafs_to_change_per_iter
        self.prop_fair_policy = PropFairness(env, resource_quantity,
                                             alloc_granularity=alloc_granularity)
        self.prop_fair_policy.initialise()
        self.last_alloc = self.prop_fair_policy.get_resource_allocation()

    def _policy_initialise(self):
        """ Initialise. """
        pass

    def _get_resource_allocation_for_loads(self, loads, *args, **kwargs):
        """ Returns the allocations for the loads. """
        try:
            recent_allocs_and_util_metrics = \
                self.performance_recorder_bank.get_recent_allocs_and_util_metrics(
                    num_recent_event_logs=self.num_recent_event_logs_for_util_computation,
                    grid_size=self.grid_size_for_util_computation)
        except Exception as e:
            logger.info('Returning fair allocation since exception %s. Returning last alloc', e)
            return self.last_alloc
        if not recent_allocs_and_util_metrics:
            next_alloc = self.last_alloc
        else:
#             last_alloc = recent_allocs_and_util_metrics[-1]['alloc']
            last_utils = recent_allocs_and_util_metrics[-1]['leaf_utils']
            next_nonint_alloc = get_next_alloc_to_try_for_maximin(
                self.last_alloc, last_utils, self.resource_quantity,
                max_num_leafs_to_change=self.max_num_leafs_to_change_per_iter)
            alloc_ratios = {key: val/self.resource_quantity
                            for key, val in next_nonint_alloc.items()}
            next_alloc = self._get_final_allocations_from_ratios(alloc_ratios)
        self.last_alloc = next_alloc
        return next_alloc

