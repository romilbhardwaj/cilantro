"""
    A policy for optimising based on some given welfare function.
    -- kirthevasank
"""

# TODO: this needs some clean up --kirthevasank

import logging
# Local
from cilantro.policies.base_policy import BasePolicy
from cilantro.policies.evo_opt import optimise_with_evo_alg
from cilantro.policies.maximin import optimise_func_maximin

logger = logging.getLogger(__name__)


class WelfarePolicy(BasePolicy):
    """ Policy for utilitarian welfare. """

    def __init__(self, env, resource_quantity, load_forecaster_bank, alloc_granularity=None,
                 num_iters_for_evo_opt=-1):
        """ Constructor.
            true_welfare_function is a function which takes allocations and loads and computes the
                welfare. To be used by the oracular policy.
        """
        super().__init__(env, resource_quantity, load_forecaster_bank, alloc_granularity)
        self.num_iters_for_evo_opt = num_iters_for_evo_opt if num_iters_for_evo_opt > 0 else \
                                     max(2000, self.env.get_num_leaf_nodes() * 100)

    def _policy_initialise(self):
        """ Initialise. """
        pass

    def _get_resource_allocation_for_loads(self, loads, *args, **kwargs):
        """ Returns the allocations for the loads. """
        func_to_optimise = self._get_function_to_optimise(loads)
        init_allocs = self._get_init_allocs(loads)
        init_allocs_and_vals = [(elem, func_to_optimise(elem)) for elem in init_allocs]
        next_alloc_non_int, _ = optimise_with_evo_alg(func_to_optimise, self.env,
                                                      self.resource_quantity,
                                                      self.num_iters_for_evo_opt,
                                                      prev_allocs_and_vals=init_allocs_and_vals)
        sum_next_alloc = sum([val for _, val in next_alloc_non_int.items()])
        next_alloc_ratios = {key: val/sum_next_alloc for key, val in next_alloc_non_int.items()}
        next_alloc = self._get_final_allocations_from_ratios(next_alloc_ratios)
        return next_alloc

    def _get_function_to_optimise(self, loads):
        """ Returns the function that needs to be optimised. """
        raise NotImplementedError('Implement in a child class.')

    @classmethod
    def _get_init_allocs(cls, loads):
        """ Returns the initial allocations. """
        # pylint: disable=unused-argument
        return []


class UtilWelfareOracularPolicy(WelfarePolicy):
    """ An oracular policy for maximising utilitarian welfare. """

    def __init__(self, env, resource_quantity, load_forecaster_bank, profiled_info_bank,
                 alloc_granularity=None, num_iters_for_evo_opt=None):
        """ Constructor. """
        super().__init__(env, resource_quantity, load_forecaster_bank=load_forecaster_bank,
                         alloc_granularity=alloc_granularity,
                         num_iters_for_evo_opt=num_iters_for_evo_opt)
        self.profiled_info_bank = profiled_info_bank

    def _get_function_to_optimise(self, loads):
        """ Returns the function to be optimised. """
        def _function_to_optimise(_env_leaf_nodes, _profiled_info_bank, _alloc, _load):
            """ Function to optimise. """
            ret = 0
            for leaf_path, leaf in _env_leaf_nodes.items():
                workload_type = leaf.get_workload_info('workload_type')
                payoff = _profiled_info_bank.get(workload_type).get_payoff_for_alloc_vals(
                    [_alloc[leaf_path]/_load[leaf_path]])[0]
                norm_util = leaf.get_norm_util_from_reward(payoff)
                ret += norm_util
            return ret
        # A function to invoke this --------------------------------------------------------------
        def _get_function_to_optimise_from_args(_env_leaf_nodes, _profiled_info_bank, _load):
            """ Returns the function to be optimised from args. """
            return lambda alloc: _function_to_optimise(_env_leaf_nodes, _profiled_info_bank, alloc,
                                                       _load)
        # return
        return _get_function_to_optimise_from_args(self.env.leaf_nodes, self.profiled_info_bank,
                                                   loads)

    def _get_init_allocs(self, loads):
        """ Returns the initial allocations. """
        return self._get_maximin_opt_allocs(loads)

    def _get_maximin_opt_allocs(self, loads):
        """ Returns the allocations from maximin opt. """
        def _maximin_function_to_optimise(_env_leaf_nodes, _profiled_info_bank, _alloc, _loads):
            """ The function to optimise. """
            ret = {}
            for leaf_path, leaf in _env_leaf_nodes.items():
                workload_type = leaf.get_workload_info('workload_type')
                payoff = _profiled_info_bank.get(workload_type).get_payoff_for_alloc_vals(
                    [_alloc[leaf_path]/_loads[leaf_path]])[0]
                ret[leaf_path] = leaf.get_norm_util_from_reward(payoff)
            return ret
        # A function to invoke this --------------------------------------------------------------
        def _get_maximin_function_to_optimise_from_args(_env_leaf_nodes, _profiled_info_bank,
                                                        _loads):
            """ Returns the function to be optimised from args. """
            return lambda alloc: _maximin_function_to_optimise(_env_leaf_nodes,
                                                               _profiled_info_bank, alloc, _loads)
        # Optimise this function with maximin -----------------------------------------------
        func_to_optimise = _get_maximin_function_to_optimise_from_args(
            self.env.leaf_nodes, self.profiled_info_bank, loads)
        return optimise_func_maximin(func_to_optimise, self.env, self.resource_quantity,
                                     self.num_iters_for_evo_opt, return_all_allocs=True)


class UtilWelfareBanditPolicy(WelfarePolicy):
    """ A bandit policy for maximising utilitarian welfare. """

    def __init__(self, env, resource_quantity, learner_bank, load_forecaster_bank,
                 alloc_granularity=None, num_iters_for_evo_opt=None):
        """ Constructor. """
        super().__init__(env, resource_quantity, load_forecaster_bank=load_forecaster_bank,
                         alloc_granularity=alloc_granularity,
                         num_iters_for_evo_opt=num_iters_for_evo_opt)
        self.learner_bank = learner_bank

    def _get_function_to_optimise(self, loads):
        """ Returns the resource allocation for the bandit policy. """
        def _function_to_optimise(_env_leaf_nodes, _learner_bank, _alloc, _load):
            """ Function to optimise. """
            ret = 0
            for leaf_path, leaf in _env_leaf_nodes.items():
                learner = _learner_bank.get(leaf_path)
                _, _, ucb = learner.compute_estimate_for_input(_alloc[leaf_path], _load[leaf_path])
                norm_util = leaf.get_norm_util_from_reward(ucb)
                ret += norm_util
            return ret
        # A function to invoke this
        def _get_function_to_optimise_from_args(_env_leaf_nodes, _learner_bank, _load):
            """ Returns the function to be optimised from args. """
            return lambda alloc: _function_to_optimise(_env_leaf_nodes, _learner_bank, alloc, _load)
        # return
        return _get_function_to_optimise_from_args(self.env.leaf_nodes, self.learner_bank, loads)


    def _get_init_allocs(self, loads):
        """ Returns the initial allocations. """
        return self._get_maximin_opt_allocs(loads)

    def _get_maximin_opt_allocs(self, loads):
        """ Returns the allocations from maximin opt. """
        def _maximin_function_to_optimise(_env_leaf_nodes, _learner_bank, _alloc, _loads):
            """ Function to optimise. """
            ret = {}
            for leaf_path, leaf in _env_leaf_nodes.items():
                learner = _learner_bank.get(leaf_path)
                _, _, ucb = learner.compute_estimate_for_input(_alloc[leaf_path], _loads[leaf_path])
                ret[leaf_path] = leaf.get_norm_util_from_reward(ucb)
            return ret
        # A function to invoke this --------------------------------------------------------------
        def _get_maximin_function_to_optimise_from_args(_env_leaf_nodes, _learner_bank,
                                                        _loads):
            """ Returns the function to be optimised from args. """
            return lambda alloc: _maximin_function_to_optimise(_env_leaf_nodes,
                                                               _learner_bank, alloc, _loads)
        # Optimise this function with maximin -----------------------------------------------
        func_to_optimise = _get_maximin_function_to_optimise_from_args(
            self.env.leaf_nodes, self.learner_bank, loads)
        return optimise_func_maximin(func_to_optimise, self.env, self.resource_quantity,
                                     self.num_iters_for_evo_opt, return_all_allocs=True)


class EgalWelfareOracularPolicy(WelfarePolicy):
    """ An oracular policy for maximising egalitarian welfare. """

    def __init__(self, env, resource_quantity, load_forecaster_bank, profiled_info_bank,
                 alloc_granularity=None, num_iters_for_evo_opt=None):
        """ Constructor. """
        super().__init__(env, resource_quantity, load_forecaster_bank=load_forecaster_bank,
                         alloc_granularity=alloc_granularity,
                         num_iters_for_evo_opt=num_iters_for_evo_opt)
        self.profiled_info_bank = profiled_info_bank

    def _get_resource_allocation_for_loads(self, loads, *args, **kwargs):
        """ Compute resource allocation for loads. """
        def _function_to_optimise(_env_leaf_nodes, _profiled_info_bank, _alloc, _loads):
            """ The function to optimise. """
            ret = {}
            for leaf_path, leaf in _env_leaf_nodes.items():
                workload_type = leaf.get_workload_info('workload_type')
                payoff = _profiled_info_bank.get(workload_type).get_payoff_for_alloc_vals(
                    [_alloc[leaf_path]/_loads[leaf_path]])[0]
                ret[leaf_path] = leaf.get_norm_util_from_reward(payoff)
            return ret
        # A function to invoke this --------------------------------------------------------------
        def _get_function_to_optimise_from_args(_env_leaf_nodes, _profiled_info_bank, _loads):
            """ Returns the function to be optimised from args. """
            return lambda alloc: _function_to_optimise(_env_leaf_nodes, _profiled_info_bank, alloc,
                                                       _loads)
        # Optimise this function with maximin -----------------------------------------------
        func_to_optimise = _get_function_to_optimise_from_args(
            self.env.leaf_nodes, self.profiled_info_bank, loads)
        return optimise_func_maximin(func_to_optimise, self.env, self.resource_quantity,
                                     self.num_iters_for_evo_opt)

    def _get_function_to_optimise(self, loads):
        """ Returns the function that needs to be optimised. """
        raise ValueError('No need to invoke this method for egalitarian policies.')


class EgalWelfareBanditPolicy(WelfarePolicy):
    """ A bandit policy for maximising egalitarian welfare. """

    def __init__(self, env, resource_quantity, learner_bank, load_forecaster_bank,
                 alloc_granularity=None, num_iters_for_evo_opt=None):
        """ Constructor. """
        super().__init__(env, resource_quantity, load_forecaster_bank=load_forecaster_bank,
                         alloc_granularity=alloc_granularity,
                         num_iters_for_evo_opt=num_iters_for_evo_opt)
        self.learner_bank = learner_bank

    def _get_resource_allocation_for_loads(self, loads, *args, **kwargs):
        """ Compute resource allocation for loads. """
        def _function_to_optimise(_env_leaf_nodes, _learner_bank, _alloc, _loads):
            """ Function to optimise. """
            ret = {}
            for leaf_path, leaf in _env_leaf_nodes.items():
                learner = _learner_bank.get(leaf_path)
                _, _, ucb = learner.compute_estimate_for_input(_alloc[leaf_path], _loads[leaf_path])
                ret[leaf_path] = leaf.get_norm_util_from_reward(ucb)
            return ret
        # A function to invoke this --------------------------------------------------------------
        def _get_function_to_optimise_from_args(_env_leaf_nodes, _learner_bank, _loads):
            """ Returns the function to be optimised from args. """
            return lambda alloc: _function_to_optimise(_env_leaf_nodes, _learner_bank, alloc,
                                                       _loads)
        # Optimise this function with maximin -----------------------------------------------
        func_to_optimise = _get_function_to_optimise_from_args(self.env.leaf_nodes,
                                                               self.learner_bank, loads)
        return optimise_func_maximin(func_to_optimise, self.env, self.resource_quantity,
                                     self.num_iters_for_evo_opt)

    def _get_function_to_optimise(self, loads):
        """ Returns the function that needs to be optimised. """
        raise ValueError('No need to invoke this method for egalitarian policies.')
