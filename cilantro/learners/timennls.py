"""
    A non-negative least squares model to compute the time taken to finish a job.
    -- kirthevasank
    -- romilbhardwaj
"""

import logging
import numpy as np
from scipy.optimize import nnls
# local
from cilantro.learners.base_learner import LearningModel


logger = logging.getLogger(__name__)


class TimeNNLS(LearningModel):
    """ A moel for learning a non-negative least squres for the time taken. """

    def __init__(self, name, int_lb, int_ub, num_bin_search_iters=30):
        """ Constructor. """
        super().__init__(name, int_lb, int_ub)
        self._scale_over_machine_features = []
        self._log_machine_features = []
        self._machine_features = []
        self._time_taken_vals = []
        self._coeffs = None
        self.num_bin_search_iters = num_bin_search_iters

    def _initialise_model_child(self):
        """ Initialise model. """
        pass

    def update_model_with_new_data(self, Allocs, Rewards, Loads, Sigmas, _):
        """ update with new data. """
        # pylint: disable=broad-except
        # pylint: disable=unused-argument
        num_data = 0
        for (alloc, rew, load) in zip(Allocs, Rewards, Loads):
            if alloc == 0:
                continue
            if rew <= 1:
                time_taken = 1 / (rew * load)
            else:
                time_taken = 1 / rew
            if not np.isfinite(time_taken):
                continue
            self._scale_over_machine_features.append(load/alloc)
            self._log_machine_features.append(np.log(alloc + 1))
            self._machine_features.append(alloc)
            # Compute time taken --------------------------------------------------
            self._time_taken_vals.append(time_taken)
            num_data += 1
        if num_data == 0:
            return
        # Now compute the NNLS estimate -----------------------------------------------------
        num_tot_data = len(self._time_taken_vals)
        ones_vector = np.ones((num_tot_data, ))
        A = np.array([ones_vector,
                      self._scale_over_machine_features,
                      self._log_machine_features,
                      self._machine_features]).transpose()
        b = np.array(self._time_taken_vals)
        try:
            self._coeffs = nnls(A, b)[0]
#             logger.info('Updated coefficients to %s.', self._coeffs)
        except Exception as e:
            logger.info('Could not update NNLS model. Exception: %s.', str(e))

    def get_recommendation(self, perf_goal, load, max_resource_amount, *args, **kwargs):
        """ Returns a recommendation. """
        # pylint: disable=arguments-differ
        if self._coeffs is None:
            logger.info('Recommendation requested when self._coeffs=%s', self._coeffs)
            return 1000.0 # Return a large value
        curr_lb = 0
        curr_ub = max_resource_amount
        # Compute target run time based on perf_goal. ---------------------------------------
        #TODO: do the throughput/latency division
        if perf_goal < 1: # then likely a latency SLO
            target_run_time = 1/(perf_goal * load)
        else: # likely a throughput SLO.
            target_run_time = 1/perf_goal
        # perform binary search to estimate the resource demand -------------------------
        for _ in range(self.num_bin_search_iters):
            curr_try = 0.5 * (curr_lb + curr_ub)
            curr_time_est = self._get_estimate_for_time(curr_try, load)
            if curr_time_est < target_run_time:
                curr_ub = curr_try
            else:
                curr_lb = curr_try
        return curr_ub

    def _get_estimate_for_time(self, alloc, load):
        """ Returns an estimate for run time. """
        return self._coeffs[0] + \
               self._coeffs[1] * load/alloc + \
               self._coeffs[2] * np.log(alloc) + \
               self._coeffs[3] * alloc

    def get_recommendation_for_upper_bound(self, perf_goal, load, *args, **kwargs):
        """ Returns a recommendation. """
        # pylint: disable=arguments-differ
        raise NotImplementedError('Not implemented yet!')

    def get_recommendation_for_lower_bound(self, perf_goal, load, *args, **kwargs):
        """ Returns a recommendation. """
        # pylint: disable=arguments-differ
        raise NotImplementedError('Not implemented yet!')

    def compute_conf_interval_for_input(self, x):
        """ Compute a confidence interval for the input point x. """
        raise NotImplementedError('Not implemented yet!')

    def compute_estimate_for_input(self, x):
        """ Obtains an estimate (along with ucbs and lcbs) for the given point. """
        raise NotImplementedError('Not implemented yet!')

