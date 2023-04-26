"""
    Policy classes for Learning demands in MMF.
    -- kirthevasank
    -- romilbhardwaj
"""

import logging
import numpy as np
# Local imports
from cilantro.policies.base_policy import BasePolicy
from cilantro.policies.mmf import HMMF

logger = logging.getLogger(__name__)


class MMFLearn(BasePolicy):
    """ A Class for MMF-Learning without strategy-proofness constraints.
    """

    def __init__(self, env, resource_quantity, load_forecaster_bank, learner_bank, *args, **kwargs):
        """ Constructor. """
        super().__init__(env, resource_quantity, load_forecaster_bank, *args, **kwargs)
        # Define variables we will need for the execution of the algorithm
        self.round_idx = None
        self.learner_bank = learner_bank
        self.hmmf_policy = HMMF(env, resource_quantity=self.resource_quantity,
                                load_forecaster_bank=load_forecaster_bank,
                                alloc_granularity=None)

    def _policy_initialise(self):
        """ Initlialisation. """
        self.hmmf_policy.initialise()

    def get_recommendations_for_unit_demands(self, _):
        """ Computes recommendations on the unit demands. """
        ret = {}
        for leaf_path, leaf in self.env.leaf_nodes.items():
            learner = self.learner_bank.get(leaf_path)
            if learner:
                rec_ub = learner.get_recommendation_for_upper_bound(leaf.threshold, 1)
                rec_mid = learner.get_recommendation(leaf.threshold, 1)
#                 rec_lb = learner.get_recommendation_for_lower_bound(leaf.threshold, 1)
                rec = 0.3 * rec_mid + 0.7 * rec_ub
                ret[leaf_path] = rec
#                 ret[leaf_path] = learner.get_recommendation_for_upper_bound(leaf.threshold, 1)
#                 ret[leaf_path] = learner.get_recommendation(leaf.threshold, 1)
#                 ret[leaf_path] = learner.get_recommendation_for_lower_bound(leaf.threshold, 1)
            else:
                known_unit_demand = leaf.unit_demand
                assert known_unit_demand is not None
                ret[leaf_path] = known_unit_demand
        logger.debug('Returning recommendations %s', str(ret))
        return ret

    def _get_resource_allocation_for_loads(self, loads):
        """ Returns the allocations for the loads. """
        # pylint: disable=too-many-locals
        # pylint: disable=arguments-differ
        # pylint: disable=too-many-branches
        # pylint: disable=broad-except
        if self.round_idx == 0:
            est_unit_demands = {leaf_path:np.inf for leaf_path in self.env.leaf_nodes}
        else:
            try:
                est_unit_demands = self.get_recommendations_for_unit_demands(loads)
            except Exception as e:
                logger.info('Unit demands could not be estimated due to exception: %s', e)
                est_unit_demands = {leaf_path:np.inf for leaf_path in self.env.leaf_nodes}
#         logger.info('-- loads %s', str(loads))
#         logger.info('-- est_unit_demands %s', str(est_unit_demands))
        # Copy the unit demands and replace them
        unit_demand_copies = []
        for leaf_path, ud_est in est_unit_demands.items():
            leaf = self.env.leaf_nodes[leaf_path]
            unit_demand_copies.append((leaf_path, leaf.unit_demand))
            leaf.set_unit_demand(ud_est)
        # Call hmmf
#         print('Requested with loads %s'%({leaf.name: val for leaf, val in loads.items()}))
        allocs_normalised = self.hmmf_policy.get_resource_allocation(loads, to_normalise=True)
        capacities = self.env.get_curr_capacities_for_all_leaf_nodes()
#         print('-- capacities', capacities)
        for leaf, val in allocs_normalised.items():
            assert val * self.resource_quantity <= capacities[leaf]
        # Compute unnormalised demands
        if self.num_alloc_quanta: # If not None, we will have to allocate in discrete quanta
            entitlements = self.env.get_entitlements()
#             print('entitlements', {key: val * self.resource_quantity for key, val in
#                                    entitlements.items()})
            # Now sort these allocations in ascending order of fair share allocations
            entitlements_as_list = list(entitlements.items())
            entitlements_as_list.sort(key=lambda x: x[1])
            num_leafs = len(entitlements_as_list)
            # Compute the discrete quanta
            ret = {}
            remainders = {}
            num_remaining_quanta = self.num_alloc_quanta
            for idx in range(num_leafs):
                leaf, leaf_entitlement = entitlements_as_list[idx]
                unnorm_alloc = allocs_normalised[leaf] * self.resource_quantity
                floor_alloc = int(np.floor(unnorm_alloc))
                ceil_alloc = int(np.ceil(unnorm_alloc))
                if ceil_alloc <= leaf_entitlement * self.resource_quantity:
                    ret[leaf] = ceil_alloc
                    diff_alloc_frac = ceil_alloc / self.resource_quantity - allocs_normalised[leaf]
                    for j in range(idx+1, num_leafs):
                        leaf_j, _ = entitlements_as_list[j]
                        allocs_normalised[leaf_j] -= diff_alloc_frac * allocs_normalised[leaf_j]
                else:
                    ret[leaf] = floor_alloc
                    remainders[leaf] = unnorm_alloc - ret[leaf]
#             print('-- allocs_normalised', allocs_normalised, self.resource_quantity,
#                    self.alloc_granularity)
#             print('-- allocs', {key: val * self.resource_quantity for key, val in
#                              allocs_normalised.items()})
#             print('-- ret', ret)
#             print('-- remainders', remainders)
            num_remaining_quanta = self.num_alloc_quanta - sum([val for _, val in ret.items()])
            # Now call randomised rounding
            if num_remaining_quanta > 0 and remainders:
                chosen_leafs = self._randomised_rounding(remainders, num_remaining_quanta)
                for leaf in chosen_leafs:
                    ret[leaf] += 1
            # Re-normalise
            ret_renorm = {key: val * self.alloc_granularity for key, val in ret.items()}
            allocs = ret_renorm
        else:
            # In this case, simply scale by resource quantity
            allocs = {key: val * self.resource_quantity for key, val in allocs_normalised.items()}
        # Before exiting, copy over the previous unit demands
        for leaf_path, ud_copy in unit_demand_copies:
            leaf = self.env.leaf_nodes[leaf_path]
            leaf.set_unit_demand(ud_copy)
#         leaf_order = sorted(list(allocs.keys()), key=lambda lf: lf.name)
#         allocs_list = [allocs[leaf] for leaf in leaf_order]
#         print('   Final-Allocations', self.round_idx, allocs_list, sum(allocs_list))
#         logger.info('MMFLearn returning allocations %s', str(allocs))
        return allocs

    def _policy_add_data_to_history(self):
        """ Updates the data and model based on feedback. """
        pass

    def _update_feedback_model(self):
        """ Updates the feedback model. """
        raise NotImplementedError('Implement in a child class.')

