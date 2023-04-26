"""
    Interleaved exploration.
    -- kirthevasank
"""

import logging
import numpy as np
# Local
from cilantro.policies.base_policy import BasePolicy
from cilantro.policies.prop_fairness import PropFairness

logger = logging.getLogger(__name__)

MIN_ALLOC_PER_LEAF = 3
DFLT_EXPL_PROB = 1/3


class MSInterleavedExploration(BasePolicy):
    """ Policy for optimizing based on upper confidence bounds. """

    def __init__(self, env, resource_quantity, load_forecaster_bank, data_logger_bank,
                 field_to_minimise, app_client_key, exploration_prob=DFLT_EXPL_PROB):
        """ Constructor. """
        super().__init__(env, resource_quantity, load_forecaster_bank, alloc_granularity=1)
        self.data_logger = data_logger_bank.get(app_client_key)
        self.app_client_key = app_client_key
        self.field_to_minimise = field_to_minimise
        self.exploration_prob = exploration_prob
        self.prop_fair_policy = PropFairness(env, resource_quantity,
                                             alloc_granularity=1)
        self.prop_fair_policy.initialise()
        self.data_logger_timestamp = None
        self.curr_best_alloc = None
        self.curr_best_val = 99999999.99

    def _policy_initialise(self):
        """ Initialise. """
        pass

    def _get_resource_allocation_for_loads(self, loads, *args, **kwargs):
        # pylint: disable=broad-except
        if self.round_idx <= 1: # Return fair allocation ----------------------------------
            return self.prop_fair_policy.get_resource_allocation()
        try:
            # First add data to EvoOpt
            new_data, new_time_stamp = \
                self.data_logger.get_data(fields=['allocs', 'p99',
                                                  'event_start_time', 'event_end_time'],
                                          start_time_stamp=self.data_logger_timestamp,
                                          end_time_stamp=None)
            num_data = len(new_data)
            if num_data > 0:
                self.data_logger_timestamp = new_time_stamp
                for elem in new_data:
                    if elem[self.field_to_minimise] < self.curr_best_val:
                        logger.info('Updating best alloc from val=%.2f to %0.2f',
                                    self.curr_best_val, elem[self.field_to_minimise])
                        self.curr_best_alloc = elem['allocs']
                        self.curr_best_val = elem[self.field_to_minimise]
            # Generate the next allocation ------------------------------------------------
            chooser = np.random.random()
            if chooser <= self.exploration_prob:
                num_leafs = len(self.env.leaf_nodes)
                env_leafs = list(self.env.leaf_nodes)
                unnorm_ratios = np.random.random((num_leafs, ))
                norm_ratios = unnorm_ratios / unnorm_ratios.sum()
                next_nonint_alloc = \
                    list(MIN_ALLOC_PER_LEAF +
                         (self.resource_quantity - MIN_ALLOC_PER_LEAF * num_leafs) * norm_ratios)
                alloc_ratios = {key: val/self.resource_quantity
                                for key, val in zip(env_leafs, next_nonint_alloc)}
                next_alloc = self._get_final_allocations_from_ratios(alloc_ratios)
                logger.info('Trying new allocation %s', next_alloc)
            else:
                # Return the current best alloc
                next_alloc = {key: int(val) for key, val in self.curr_best_alloc.items()}
                logger.info('Using current best allocation %s', next_alloc)
        except Exception as e:
            # Using an except here since EvoAlg policies are failing early on in the experiment,
            # probably because there isn't much data in the data loggers.
            logger.info('MSILE is returning fair share as exception occurred: %s', str(e))
            next_alloc = self.prop_fair_policy.get_resource_allocation()
        return next_alloc

