"""
    Implements multiplicative increase, additive decrease to set the demands of the users.
    -- kirthevasank
    -- romilbhardwaj
"""


from copy import deepcopy
import logging
# Local imports
from cilantro.policies.mmflearn import MMFLearn

logger = logging.getLogger(__name__)


INCREASE_FACTOR = 5
DECREASE_DELTA = 1


class MultIncAddDec(MMFLearn):
    """ MultIncAddDec. """

    def __init__(self, env, resource_quantity, load_forecaster_bank, reward_forecaster_bank,
                 *args, **kwargs):
        """ Constructor. """
        super().__init__(env, resource_quantity, load_forecaster_bank, learner_bank=None,
                         *args, **kwargs)
        # Define variables
        self.reward_forecaster_bank = reward_forecaster_bank
        self.last_allocs = None

    def _get_recommendations_for_unit_demands(self, loads):
        """ Returns the allocation for the loads. """
        if not self.last_allocs:
            # Provide demands which guarantees that each user gets at least their fair share.
            entitlements = self.env.get_entitlements()
            ret = {leaf: val * 2 / loads[leaf] for leaf, val in entitlements.items()}
        else:
            to_upsize = []
            to_downsize = []
            ret_demands = deepcopy(self.last_allocs)
            for leaf_path, leaf in self.env.leaf_nodes.items():
                reward_forecaster = self.reward_forecaster_bank.get(leaf_path)
                if reward_forecaster is None:
                    reward_est = 1
                else:
                    _, reward_est, _ = reward_forecaster.forecast()
                if reward_est < leaf.threshold:
                    to_upsize.append(leaf_path)
                elif reward_est >= leaf.threshold:
                    to_downsize.append(leaf_path)
            # Do the up/down-sizing
            for leaf_path in to_upsize:
                ret_demands[leaf_path] *= INCREASE_FACTOR
            for leaf_path in to_downsize:
                ret_demands[leaf_path] -= DECREASE_DELTA
            # Now determine the allocation
            ret = {leaf: val/loads[leaf] for leaf, val in entitlements.items()}
        return ret

    def _get_resource_allocation_for_loads(self, loads):
        """ Returns the allocations for the loads. """
        ret = super()._get_resource_allocation_for_loads(loads)
        self.last_allocs = deepcopy(ret)
        return ret

    def _policy_add_data_to_history(self):
        """ Updates the data and model based on feedback. """
        pass

    def _update_feedback_model(self):
        """ Updates the feedback model. """
        raise NotImplementedError('Implement in a child class.')


