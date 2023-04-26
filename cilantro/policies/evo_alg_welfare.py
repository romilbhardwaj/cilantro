"""
    Implements evolutionary algorithms for resource allocation.
    -- kirthevasank
    -- romilbhardwaj
"""

import logging
# Local
from cilantro.policies.base_policy import BasePolicy
from cilantro.policies.evo_opt import EvoOpt
from cilantro.policies.prop_fairness import PropFairness

logger = logging.getLogger(__name__)


class WelfareEvoAlg(BasePolicy):
    """ A policy for maximising welfare using an evolutionary algorithm. """

    def __init__(self, env, resource_quantity, performance_recorder_bank, load_forecaster_bank,
                 alloc_granularity=None, num_mutations_per_epoch=3,
                 num_recent_event_logs_for_util_computation=20,
                 grid_size_for_util_computation=5):
        """ Constructor. """
        super().__init__(env, resource_quantity, load_forecaster_bank, alloc_granularity)
        self.performance_recorder_bank = performance_recorder_bank
        self.num_recent_event_logs_for_util_computation = num_recent_event_logs_for_util_computation
        self.grid_size_for_util_computation = grid_size_for_util_computation
        self.key_for_welfare = self._get_key_for_welfare()
        self.evo_opt = EvoOpt(env, resource_quantity,
                              num_mutations_per_epoch=num_mutations_per_epoch)
        self.prop_fair_policy = PropFairness(env, resource_quantity,
                                             alloc_granularity=alloc_granularity)
        self.prop_fair_policy.initialise()

    def _policy_initialise(self):
        """ Initialise. """
        pass

    @classmethod
    def _get_key_for_welfare(cls):
        """ Returns the key for the utility. """
        raise NotImplementedError('Implement in a child class!')

    def _get_resource_allocation_for_loads(self, loads, *args, **kwargs):
        """ Returns the allocations for the loads. """
        # pylint: disable=broad-except
        if self.round_idx == 0: # Return fair allocation on the first round ------------------------
            return self.prop_fair_policy.get_resource_allocation()
        try:
            recent_allocs_and_util_metrics = \
                self.performance_recorder_bank.get_recent_allocs_and_util_metrics(
                    num_recent_event_logs=self.num_recent_event_logs_for_util_computation,
                    grid_size=self.grid_size_for_util_computation)
            if not recent_allocs_and_util_metrics:
                next_alloc = self.prop_fair_policy.get_resource_allocation()
            else:
                allocs = [elem['alloc'] for elem in recent_allocs_and_util_metrics]
                welfares = [elem[self.key_for_welfare] for elem in recent_allocs_and_util_metrics]
                self.evo_opt.add_data(allocs, welfares)
                # Generate the next allocation ----------------------------------------------
                next_nonint_alloc = self.evo_opt.get_next_eval_point()
                alloc_ratios = {key: val/self.resource_quantity
                                for key, val in next_nonint_alloc.items()}
                next_alloc = self._get_final_allocations_from_ratios(alloc_ratios)
        except Exception as e:
            # Using an except here since EvoAlg policies are failing early on in the experiment,
            # probably because there isn't much data in the data loggers.
            logger.info('EvoAlg is returning fair share as exception occurred: %s', str(e))
            next_alloc = self.prop_fair_policy.get_resource_allocation()
        return next_alloc


class UtilWelfareEvoAlg(WelfareEvoAlg):
    """ Evolutionary algorithm for utilitarian welfare. """

    @classmethod
    def _get_key_for_welfare(cls):
        """ Returns the key for the utility. """
        return 'util_welfare'


class EgalWelfareEvoAlg(WelfareEvoAlg):
    """ Evolutionary algorithm for utilitarian welfare. """

    @classmethod
    def _get_key_for_welfare(cls):
        """ Returns the key for the utility. """
        return 'egal_welfare'

