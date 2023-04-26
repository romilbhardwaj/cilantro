"""
    Evolutionary optimization for the microservices policy.
    -- kirthevasank
    -- romilbhardwaj
"""

import logging
# Local
from cilantro.policies.base_policy import BasePolicy
from cilantro.policies.prop_fairness import PropFairness
from cilantro.policies.evo_opt import get_dflt_mutation_op, EvoOpt

logger = logging.getLogger(__name__)

MIN_ALLOC_PER_LEAF = 3


class MSEvoOpt(BasePolicy):
    """ Policy for optimizing based on upper confidence bounds. """

    def __init__(self, env, resource_quantity, load_forecaster_bank, data_logger_bank,
                 field_to_minimise, app_client_key):
        """ Constructor. """
        super().__init__(env, resource_quantity, load_forecaster_bank, alloc_granularity=1)
        self.data_logger = data_logger_bank.get(app_client_key)
        self.app_client_key = app_client_key
        mutation_op = get_dflt_mutation_op(min_num_steps=1, max_num_steps=5,
                                           min_alloc_per_leaf=MIN_ALLOC_PER_LEAF)
        self.evo_opt = EvoOpt(env, resource_quantity,
                              mutation_op=mutation_op)
        self.field_to_minimise = field_to_minimise
        self.prop_fair_policy = PropFairness(env, resource_quantity,
                                             alloc_granularity=1)
        self.prop_fair_policy.initialise()
        self.data_logger_timestamp = None
        self.total_data_added = 0

    def _policy_initialise(self):
        """ Initialise. """
        pass

    def _get_resource_allocation_for_loads(self, loads, *args, **kwargs):
        # pylint: disable=broad-except
        if self.round_idx == 0: # Return fair allocation on the first round ------------------------
            return self.prop_fair_policy.get_resource_allocation()
        # First add data to EvoOpt
        new_data, new_time_stamp = \
            self.data_logger.get_data(fields=['allocs', 'p99',
                                              'event_start_time', 'event_end_time'],
                                      start_time_stamp=self.data_logger_timestamp,
                                      end_time_stamp=None)
        num_data = len(new_data)
        if num_data > 0:
            self.total_data_added += num_data
            self.data_logger_timestamp = new_time_stamp
            data_X = [elem['allocs'] for elem in new_data]
            data_Y = [-elem[self.field_to_minimise] for elem in new_data]
            self.evo_opt.add_data(data_X, data_Y)
        # Generate the next allocation -------------------------------------------------------------
        if self.total_data_added <= 0:
            next_alloc = self.prop_fair_policy.get_resource_allocation()
        else:
            next_nonint_alloc = self.evo_opt.get_next_eval_point()
            alloc_ratios = {key: val/self.resource_quantity
                            for key, val in next_nonint_alloc.items()}
            next_alloc = self._get_final_allocations_from_ratios(alloc_ratios)
        return next_alloc
#         try:
#             # First add data to EvoOpt
#             all_data = self.data_logger.history[self.last_history_idx: ]
#             num_data = len(all_data)
#             self.last_history_idx += num_data
#             if num_data > 0:
#                 data_X = [elem['allocs'] for elem in all_data]
#                 data_Y = [-elem[self.field_to_minimise] for elem in all_data]
#                 self.evo_opt.add_data(data_X, data_Y)
#             # Generate the next allocation ----------------------------------------------
#             next_nonint_alloc = self.evo_opt.get_next_eval_point()
#             alloc_ratios = {key: val/self.resource_quantity
#                             for key, val in next_nonint_alloc.items()}
#             next_alloc = self._get_final_allocations_from_ratios(alloc_ratios)
#         except Exception as e:
#             # Using an except here since EvoAlg policies are failing early on in the experiment,
#             # probably because there isn't much data in the data loggers.
#             logger.info('EvoAlg is returning fair share as exception occurred: %s', str(e))
#             next_alloc = self.prop_fair_policy.get_resource_allocation()
#         return next_alloc

