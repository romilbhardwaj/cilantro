"""
    Harness for autoscaling from learned utilities.
    -- kirthevasank
"""

import logging
import numpy as np
# Local
from cilantro.policies.base_policy import BasePolicy

logger = logging.getLogger(__name__)


class AutoScalingBasePolicy(BasePolicy):
    """ Base class for autoscaling. """

    def __init__(self, env, resource_quantity, load_forecaster_bank=None):
        """ Constructor. """
        # Setting alloc_granularity to 1 by default -----------------------------
        super().__init__(env, resource_quantity, load_forecaster_bank, alloc_granularity=1)

    def _policy_initialise(self):
        """ Initialisation in a child class. """
        pass

    def _get_autoscaling_resource_allocation_for_loads(self, load, *args, **kwargs):
        """ Get autoscaling resource allocation. """
        raise NotImplementedError('Implement in a child class')

    def _get_resource_allocation_for_loads(self, loads, *args, **kwargs):
        """ Obtain resource allocation for loads. """
        # pylint: disable=arguments-differ
        allocs = self._get_autoscaling_resource_allocation_for_loads(loads, *args, **kwargs)
        tot_alloc = sum([val for _, val in allocs.items()])
        if tot_alloc > self.resource_quantity:
            alloc_ratios = {key: val/tot_alloc for key, val in allocs.items()}
            allocs = self._get_final_allocations_from_ratios(alloc_ratios)
            logger.info(
                'Estimated resource demand is %d, but there are only %d resources.Returning %s.',
                tot_alloc, self.resource_quantity, allocs)
        return allocs


class SLOAutoScaler(AutoScalingBasePolicy):
    """ Autoscaling based on SLOs. """

    def get_unit_demand_estimates(self, est_type):
        """ Returns a dictionary of the form {leaf_path: (est, lcb, ucb)} where leaf_path is the
            path to the leaf and (est, lcb, ucb) are th estimates and lower and upper confidence
            bounds for the function.
        """
        raise NotImplementedError('Implement in a child class')

    def _get_autoscaling_resource_allocation_for_loads(self, loads, ud_est_type='lcb',
                                                       *args, **kwargs):
        """ Obtain resource allocation for loads. """
        # pylint: disable=arguments-differ
        unit_demands = self.get_unit_demand_estimates(est_type=ud_est_type)
        allocs = {}
        for leaf_path, ud in unit_demands.items():
            allocs[leaf_path] = int(np.ceil(ud * loads[leaf_path]))
        return allocs


class OracularAutoScaler(SLOAutoScaler):
    """ Oracular auto scaler. """

    def __init__(self, env, resource_quantity, load_forecaster_bank, profiled_info_bank):
        """ Constructor. """
        # Setting alloc_granularity to 1 by default -----------------------------
        super().__init__(env, resource_quantity, load_forecaster_bank)
        self.profiled_info_bank = profiled_info_bank
        self.unit_demand_etimates_type_dict = {}

    def get_unit_demand_estimates(self, est_type):
        """ Obtain unit demand est type. """
        if not (est_type in self.unit_demand_etimates_type_dict):
            # If not available, compute it
            ret = {leaf_path: self.profiled_info_bank.get_unit_demand_for_leaf_node(leaf_node)
                   for leaf_path, leaf_node in self.env.leaf_nodes.items()}
            self.unit_demand_etimates_type_dict[est_type] = ret
        return self.unit_demand_etimates_type_dict[est_type]


class BanditAutoScaler(SLOAutoScaler):
    """ Bandit auto scaler. """

    def __init__(self, env, resource_quantity, load_forecaster_bank, learner_bank):
        """ Constructor. """
        # Setting alloc_granularity to 1 by default -----------------------------
        super().__init__(env, resource_quantity, load_forecaster_bank)
        self.learner_bank = learner_bank

    def get_unit_demand_estimates(self, est_type):
        """ Obtain unit demand est type. """
        ret_probs = [0.1, 0.9, 1.0]
        ret = {}
        for leaf_path, leaf_node in self.env.leaf_nodes.items():
            learner = self.learner_bank.get(leaf_path)
            chooser = np.random.random()
            if chooser <= ret_probs[0]:
                ret[leaf_path] = learner.get_recommendation_for_lower_bound(leaf_node.threshold, 1)
            elif chooser <= ret_probs[1]:
                ret[leaf_path] = learner.get_recommendation(leaf_node.threshold, 1)
            else:
                ret[leaf_path] = learner.get_recommendation_for_upper_bound(leaf_node.threshold, 1)
#             if est_type == 'est':
#                 ret[leaf_path] = learner.get_recommendation(leaf_node.threshold, 1)
#             elif est_type == 'ucb':
#                 ret[leaf_path] = learner.get_recommendation_for_upper_bound(leaf_node.threshold,
#                                                                             1)
#             elif est_type == 'lcb':
#                 ret[leaf_path] = learner.get_recommendation_for_lower_bound(leaf_node.threshold,
#                                                                             1)
#             else:
#                 raise ValueError('Unknown est_type %s'%(est_type))
        return ret

