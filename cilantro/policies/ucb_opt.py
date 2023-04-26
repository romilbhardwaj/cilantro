"""
    Bandit algorithm which maximises an upper confidence bound.
    To be used in settings where we wish to maximise a single criteria computed for the whole system
    (as opposed to fairness based criteria).
    -- kirthevasank
"""

import logging
import numpy as np
# Local
from cilantro.policies.base_policy import BasePolicy
from cilantro.policies.evo_opt import optimise_with_evo_alg


logger = logging.getLogger(__name__)

MIN_ALLOC_PER_LEAF = 4


class UCBOpt(BasePolicy):
    """ Policy for optimizing based on upper confidence bounds. """

    def __init__(self, env, resource_quantity, learner_bank, load_forecaster_bank, app_client_key,
                 num_iters_for_evo_opt):
        """ Constructor. """
        super().__init__(env, resource_quantity, load_forecaster_bank, alloc_granularity=1)
        self.learner = learner_bank.get(app_client_key)
        self.load_forecaster = load_forecaster_bank.get(app_client_key)
        self.app_client_key = app_client_key
        self.num_iters_for_evo_opt = num_iters_for_evo_opt

    def _policy_initialise(self):
        """ Initialise policy. """
        pass

    def _get_resource_allocation_for_loads(self, loads, *args, **kwargs):
        """ Obtains the allocation. """
        if self.round_idx < 3:
            # Return random allocations -------------------------------------------------
            num_leafs = len(self.env.leaf_nodes)
            if self.round_idx <= 1:
                unnorm_ratios = np.ones((num_leafs, ))
            else:
                unnorm_ratios = np.random.random((num_leafs, ))
            ratios = list(MIN_ALLOC_PER_LEAF/self.resource_quantity + \
                          (1 - MIN_ALLOC_PER_LEAF * num_leafs/self.resource_quantity) * \
                          (unnorm_ratios / unnorm_ratios.sum()))
            alloc_ratios = dict(zip(self.env.leaf_nodes.keys(), ratios))
            ret = self._get_final_allocations_from_ratios(alloc_ratios, min_alloc_quanta=1)
            ret = {key: int(val) for key, val in ret.items()}
#             logger.info('Returning random allocation %s.', ret)
            return ret
        else:
            # Maximise an upper confidence bound -------------------------------------------
            def _get_ucb(_alloc, _load, _beta_t):
                """ Computes the UCB. """
                mean_pred, std_pred = self.learner.get_mean_pred_and_std_for_alloc_load(
                    _alloc, _load)
                return mean_pred + _beta_t * std_pred
            def _get_func_to_maximise(_load, _beta_t=2.5):
                """ Returns the function to be maximised. """
                return lambda alloc: _get_ucb(alloc, _load, _beta_t)
            func_to_maximise = _get_func_to_maximise(loads[self.app_client_key])
            next_alloc_non_int, _ = optimise_with_evo_alg(func_to_maximise, self.env,
                                                          self.resource_quantity,
                                                          self.num_iters_for_evo_opt,
                                                          prev_allocs_and_vals=[],
                                                          min_alloc_per_leaf=MIN_ALLOC_PER_LEAF)
            sum_next_alloc = sum([val for _, val in next_alloc_non_int.items()])
            next_alloc_ratios = {key: val/sum_next_alloc for key, val in next_alloc_non_int.items()}
            next_alloc = self._get_final_allocations_from_ratios(next_alloc_ratios)
            return next_alloc

