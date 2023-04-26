"""
    Implements Quasar.
    -- kirthevasank
"""

import bisect
import logging
import numpy as np
# Local imports
from cilantro.policies.mmflearn import MMFLearn

logger = logging.getLogger(__name__)


class Quasar(MMFLearn):
    """ Quasar """

    def __init__(self, env, resource_quantity, performance_recorder_bank, load_forecaster_bank,
                 alloc_granularity, num_recent_event_logs_for_util_computation=200,
                 grid_size_for_util_computation=10, max_num_leafs_to_change_per_iter=1,
                 learning_rate=0.0005, regul_factor=0.00005, max_num_pq_iters=4000, pq_err_tol=0.01,
                 num_init_alloc_rounds=2, mat_rank_approx=3):
        """ Constructor. """
        # pylint: disable=too-many-arguments
        super().__init__(env, resource_quantity, load_forecaster_bank, learner_bank=None,
                         alloc_granularity=alloc_granularity)
        self.performance_recorder_bank = performance_recorder_bank
        # For util computation --------------------------------------------------------------------
        self.num_recent_event_logs_for_util_computation = num_recent_event_logs_for_util_computation
        self.grid_size_for_util_computation = grid_size_for_util_computation
        self.max_num_leafs_to_change_per_iter = max_num_leafs_to_change_per_iter
        # Policy hyperparameters ------------------------------------------------------------------
        self.learning_rate = learning_rate
        self.regul_factor = regul_factor
        self.max_num_pq_iters = max_num_pq_iters
        self.pq_err_tol = pq_err_tol
        self.mat_rank_approx = mat_rank_approx
        self.num_init_alloc_rounds = num_init_alloc_rounds
        # Quantities we will need for updating the algorithm ---------------------------------------
        self.max_res_amount = int(self.resource_quantity // 3)
        self.workload_type_dict = {}
        self.workload_types = []
        self.wltype_utils_dict = {}
        self.wltype_num_allocs_dict = {}
        self.num_workload_types = None
        self.last_rec = None

    def _policy_initialise(self):
        """ Initlialisation. """
        super()._policy_initialise()
        for leaf_path, leaf_node in self.env.leaf_nodes.items():
            workload_type = leaf_node.get_workload_info('workload_type')
            if not (workload_type in self.workload_type_dict):
                self.workload_type_dict[workload_type] = []
                self.workload_types.append(workload_type)
            self.workload_type_dict[workload_type].append(leaf_path)
        self.wltype_utils_dict = {key: [0] * self.max_res_amount for key in self.workload_types}
        self.wltype_num_allocs_dict = {key: [0] * self.max_res_amount for key in
                                       self.workload_types}
        self.num_workload_types = len(self.workload_types)
        self.mat_rank_approx = max(1, min(self.mat_rank_approx, self.num_workload_types//2))

    @classmethod
    def _get_err(cls, U, Q, P):
        """ Returns the error. """
        nonzero_U = U != 0
        R = np.dot(Q, P)
        err = nonzero_U * (U - R)
        return err, np.linalg.norm(err, 'fro')

    def _PQ_recon_iter(self, U, Q, P):
        """ Performs one iteration of PQ reconstruction. """
        err, err_norm = self._get_err(U, Q, P)
#         print('mat shapes', U.shape, Q.shape, P.shape, err.shape)
#         print(U)
#         print(Q)
#         print(P)
#         print(err)
        Q = (Q + self.learning_rate * (np.dot(err, P.transpose()) - self.regul_factor * Q)
             ).clip(min=0)
        P = (P + self.learning_rate * (np.dot(Q.transpose(), err) - self.regul_factor * P)
             ).clip(min=0)
        return Q, P, err_norm

    def _PQ_reconstuction(self, U):
        """ Performs PQ reconstruction. """
        Q = np.random.random(size=(self.num_workload_types, self.mat_rank_approx))
        P = np.random.random(size=(self.mat_rank_approx, self.max_res_amount))
        U_norm = np.linalg.norm(U, ord='fro')
        _, init_err_norm = self._get_err(U, Q, P)
        for pq_iter in range(self.max_num_pq_iters):
            Q, P, err_norm = self._PQ_recon_iter(U, Q, P)
            if err_norm <= U_norm * self.pq_err_tol:
                break
        logger.info('PQ-recon terminated after %d iters. init_err=%0.4f, final_err=%0.4f',
                    pq_iter, init_err_norm, err_norm)
        if not np.isfinite(err_norm):
            return None
        R = (np.dot(Q, P)).clip(min=0, max=1)
        # Create a dictionary -------------------------------------------------------------
        ret = {}
        for idx, wl_type in enumerate(self.workload_types):
            curr_vals = R[idx, :]
            curr_vals = np.maximum.accumulate(curr_vals) # correct for monotonicity
            ret[wl_type] = list(curr_vals)
        return ret

    def get_recommendations_for_unit_demands(self, loads):
        """ Obtains recommendations for the unit demand. Over-rides the implementation in
            mmlearn.py
        """
        # First update the utility values ----------------------------------------------------
        try:
            recent_allocs_and_util_metrics = \
                self.performance_recorder_bank.get_recent_allocs_and_util_metrics(
                    num_recent_event_logs=self.num_recent_event_logs_for_util_computation,
                    grid_size=self.grid_size_for_util_computation)
        except Exception as e:
            logger.info('Returning large demands since exception %s.', e)
            return {leaf_path: 10000 for leaf_path in self.env.leaf_nodes}
        if recent_allocs_and_util_metrics is None:
            logger.info('Returning large demands since recent_metrics was None.')
            return {leaf_path: 10000 for leaf_path in self.env.leaf_nodes}
        for leaf_path, leaf_node in self.env.leaf_nodes.items():
            workload_type = leaf_node.get_workload_info('workload_type')
            for result in recent_allocs_and_util_metrics:
                alloc = int(np.round(result['alloc'][leaf_path]))
                if 1 <= alloc <= self.max_res_amount:
                    rew = result['leaf_rewards'][leaf_path]
                    self.wltype_utils_dict[workload_type][alloc - 1] += rew
                    self.wltype_num_allocs_dict[workload_type][alloc - 1] += 1
        # Construct the U matrix -------------------------------------------------------
        U = np.zeros((self.num_workload_types, self.max_res_amount))
        for wl_idx, wl_type in enumerate(self.workload_types):
            for res_idx in range(self.max_res_amount):
                if self.wltype_num_allocs_dict[wl_type][res_idx] > 0:
                    U[wl_idx, res_idx] = self.wltype_utils_dict[wl_type][res_idx] / \
                                         self.wltype_num_allocs_dict[wl_type][res_idx]
        # PQ low rank reconstruction  --------------------------------------------------------
        est_rewards_for_wls = self._PQ_reconstuction(U)
        if est_rewards_for_wls is None:
            return self.last_rec
        ret = {}
        true_demands = {}
        for leaf_path, leaf_node in self.env.leaf_nodes.items():
            workload_type = leaf_node.get_workload_info('workload_type')
            curr_rew_list = est_rewards_for_wls[workload_type]
            leaf_demand = bisect.bisect_right(curr_rew_list, leaf_node.threshold) + 1
            true_demands[leaf_path] = leaf_demand
            ret[leaf_path] = leaf_demand / loads[leaf_path]
        logger.info('Quasar returning demands: %s', true_demands)
        return ret

    def _get_random_allocation(self):
        """ Returns a random allocation. """
        num_leafs = len(self.env.leaf_nodes)
        unnorm_ratios = np.random.random((num_leafs, ))
        ratios = list(1/self.resource_quantity + \
                      (1 - 1/self.resource_quantity) * (unnorm_ratios / unnorm_ratios.sum()))
        alloc_ratios = dict(zip(self.env.leaf_nodes.keys(), ratios))
        ret = self._get_final_allocations_from_ratios(alloc_ratios, min_alloc_quanta=1)
        ret = {key: int(val) for key, val in ret.items()}
        self.last_rec = ret
        return ret

    def _get_resource_allocation_for_loads(self, loads):
        """ Returns the allocations for the loads. """
        # pylint: disable=broad-except
        if self.round_idx < self.num_init_alloc_rounds:
            return self._get_random_allocation()
        try:
            ret = super()._get_resource_allocation_for_loads(loads)
            self.last_ret = ret
            return ret
        except Exception as e:
            logger.info('Received exception %s. Returning random allocation on round %d',
                        str(e), self.round_idx)
            return self._get_random_allocation()

    def _policy_add_data_to_history(self):
        """ Updates the data and model based on feedback. """
        pass

    def _update_feedback_model(self):
        """ Updates the feedback model. """
        raise NotImplementedError('Implement in a child class.')

