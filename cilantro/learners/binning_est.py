"""
    A simple binning estimator.
    --kirthevasank
    -- romilbhardwaj
"""

import logging
import numpy as np
# local
from cilantro.learners.base_learner import LearningModel


logger = logging.getLogger(__name__)


class BinningEst(LearningModel):
    """ A moel for learning a non-negative least squres for the time taken. """

    def __init__(self, name, int_lb, int_ub, glob_lower_bound, glob_upper_bound, num_bins=30):
        """ Constructor. """
        super().__init__(name, int_lb, int_ub)
        self.num_bins = num_bins
        self.bin_grid = np.linspace(int_lb, int_ub, num_bins + 1)
        self.y_in_bins = [[] for _ in range(num_bins)]
        self.x_in_bins = [[] for _ in range(num_bins)]
        self.bin_means = [None for _ in range(num_bins)]
        self.bin_stds = [None for _ in range(num_bins)]
        self.bin_stderrs = [None for _ in range(num_bins)]
        self.glob_lower_bound = glob_lower_bound
        self.glob_upper_bound = glob_upper_bound
        self.bin_lcbs = [glob_lower_bound for _ in range(num_bins)]
        self.bin_ucbs = [glob_upper_bound for _ in range(num_bins)]
        self.bin_lcbs_dist = [glob_lower_bound for _ in range(num_bins)]
        self.bin_ucbs_dist = [glob_upper_bound for _ in range(num_bins)]

    def _initialise_model_child(self):
        """ Initialise model. """
        pass

    def _update_model(self):
        """ Updates model. """
        for idx, bin_data in enumerate(self.y_in_bins):
            if len(bin_data) < 2:
                curr_bin_mean = None
                curr_bin_std = None
                curr_bin_stderr = None
            else:
                curr_bin_mean = np.mean(bin_data)
                curr_bin_std = np.std(bin_data)
                curr_bin_stderr = curr_bin_std / np.sqrt(len(bin_data))
            self.bin_means[idx] = curr_bin_mean
            self.bin_stds[idx] = curr_bin_std
            self.bin_stderrs[idx] = curr_bin_stderr
        dist_std_coeff = 2
        std_coeff = 2
        for idx, bin_data in enumerate(self.y_in_bins):
            if len(bin_data) < 2:
                self.bin_lcbs[idx] = self.glob_lower_bound if idx == 0 else self.bin_lcbs[idx-1]
                self.bin_lcbs_dist[idx] = \
                    self.glob_lower_bound if idx == 0 else self.bin_lcbs_dist[idx-1]
            else:
                prev_min = self.glob_lower_bound if idx == 0 else self.bin_lcbs[idx-1]
                self.bin_lcbs[idx] = max(self.bin_lcbs[idx], prev_min,
                                         self.bin_means[idx] - std_coeff * self.bin_stderrs[idx])
                prev_min_dist = self.glob_lower_bound if idx == 0 else self.bin_lcbs_dist[idx-1]
                self.bin_lcbs_dist[idx] = max(self.bin_lcbs_dist[idx], prev_min_dist,
                                              self.bin_means[idx] - dist_std_coeff * self.bin_stds[idx])
        for idx, bin_data in reversed(list(enumerate(self.y_in_bins))):
            if len(bin_data) < 2:
                self.bin_ucbs[idx] = self.glob_upper_bound if idx == self.num_bins - 1 else \
                                     self.bin_ucbs[idx+1]
                self.bin_ucbs_dist[idx] = self.glob_upper_bound if idx == self.num_bins - 1 else \
                                          self.bin_ucbs_dist[idx+1]
            else:
                prev_max = self.glob_upper_bound if idx == self.num_bins - 1 \
                           else self.bin_ucbs[idx+1]
                self.bin_ucbs[idx] = min(self.bin_ucbs[idx], prev_max,
                                         self.bin_means[idx] + std_coeff * self.bin_stderrs[idx])
                prev_max_dist = self.glob_upper_bound if idx == self.num_bins - 1 \
                                else self.bin_ucbs_dist[idx+1]
                self.bin_ucbs_dist[idx] = min(self.bin_ucbs_dist[idx], prev_max_dist,
                                              self.bin_means[idx] + dist_std_coeff * self.bin_stds[idx])

    def update_model_with_new_data(self, Allocs, Rewards, Loads, Sigmas, _):
        """ update with new data. """
        X = [alloc/load for alloc, load in zip(Allocs, Loads)]
        self.add_multiple_data_points(X, Rewards, Sigmas)

    def get_sigma_estimate(self):
        """ Returns an estimate for sigma. """
        return np.mean([elem for elem in self.bin_stds if elem is not None])

    def add_multiple_data_points(self, X, Y, _):
        """ Adds multiple data points. """
        bin_ids = np.digitize(X, self.bin_grid)
        for idx, x in enumerate(X):
            if x <= 0:
                continue
            bidx = bin_ids[idx]
            if 1 <= bidx <= self.num_bins:
                store_idx = bidx - 1
                self.x_in_bins[store_idx].append(x)
                self.y_in_bins[store_idx].append(Y[idx])
        self._update_model()

    # Utilities used for inference -----------------------------------------------------------------
    def compute_conf_interval_for_input(self, x):
        """ Obtains a confidence interval for the given point. """
        curr_bin_idx = min(np.digitize(x, self.bin_grid), self.num_bins) - 1
        return self.bin_lcbs[curr_bin_idx], self.bin_ucbs[curr_bin_idx]

    def compute_estimate_for_input(self, x):
        """ Obtains an estimate (along with ucbs and lcbs) for the given point. """
        lcb, ucb = self.compute_conf_interval_for_input(x)
        est = (lcb + ucb) / 2
        return est, lcb, ucb

    def compute_dist_interval_for_input(self, x):
        """ Obtains a confidence interval for the given point. """
        curr_bin_idx = min(np.digitize(x, self.bin_grid), self.num_bins) - 1
        return self.bin_lcbs_dist[curr_bin_idx], self.bin_ucbs_dist[curr_bin_idx]

    def compute_dist_estimate_for_input(self, x):
        """ Obtains an estimate (along with ucbs and lcbs) for the given point. """
        lcb, ucb = self.compute_dist_interval_for_input(x)
#         width = (ucb - lcb) / 2
#         est = 
#         est = (lcb + ucb) / 2
        est = 0.6 * lcb + 0.4 * ucb
#         est = 
        return est, lcb, ucb

    # Recommendations ------------------------------------------------------------------------------
    def get_recommendation(self, perf_goal, *args, **kwargs):
        """ Obtain recommendation. """
        raise NotImplementedError('Implement in a child class.')

    def get_recommendation_for_upper_bound(self, perf_goal, *args, **kwargs):
        """ Obtain recommendation. """
        raise NotImplementedError('Implement in a child class.')

    def get_recommendation_for_lower_bound(self, perf_goal, *args, **kwargs):
        """ Obtain recommendation. """
        raise NotImplementedError('Implement in a child class.')

