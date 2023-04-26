"""
    Implements Ernest.
    -- kirthevasank
"""

import logging
import numpy as np
# Local imports
from cilantro.learners.timennls import TimeNNLS
from cilantro.policies.mmflearn import MMFLearn

logger = logging.getLogger(__name__)


class Ernest(MMFLearn):
    """ Ernest. """

    def __init__(self, env, resource_quantity, load_forecaster_bank, learner_bank, *args, **kwargs):
        """ Constructor. """
        super().__init__(env, resource_quantity, load_forecaster_bank, learner_bank,
                         *args, **kwargs)
        for _, learner in self.learner_bank.enumerate():
            assert isinstance(learner.get_model_for_serving(), TimeNNLS)

    def get_recommendations_for_unit_demands(self, loads):
        """ Obtains recommendations for the unit demand. Over-rides the implementation in
            mmlearn.py
        """
        ret = {}
        for leaf_path, leaf in self.env.leaf_nodes.items():
            learner = self.learner_bank.get(leaf_path)
            if learner:
                curr_demand = learner.get_model_for_serving().get_recommendation(
                    perf_goal=leaf.threshold, load=loads[leaf_path],
                    max_resource_amount=self.resource_quantity)
                ret[leaf_path] = curr_demand/loads[leaf_path]
#         logger.info('[Ernest] Returning demands: %s', ret)
        return ret

    def _get_resource_allocation_for_loads(self, loads):
        """ Returns the allocations for the loads. """
        if self.round_idx < 10:
            # Return random allocations -------------------------------------------------
            num_leafs = len(self.env.leaf_nodes)
            unnorm_ratios = np.random.random((num_leafs, ))
            ratios = list(1/self.resource_quantity + \
                          (1 - 1/self.resource_quantity) * (unnorm_ratios / unnorm_ratios.sum()))
            alloc_ratios = dict(zip(self.env.leaf_nodes.keys(), ratios))
            ret = self._get_final_allocations_from_ratios(alloc_ratios, min_alloc_quanta=1)
            ret = {key: int(val) for key, val in ret.items()}
#             logger.info('Returning random allocation %s.', ret)
            return ret
        else:
#             logger.info('Returning MMF allocation')
            return super()._get_resource_allocation_for_loads(loads)


    def _policy_add_data_to_history(self):
        """ Updates the data and model based on feedback. """
        pass

    def _update_feedback_model(self):
        """ Updates the feedback model. """
        raise NotImplementedError('Implement in a child class.')

