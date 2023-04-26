"""
    This model simply records, i.e. 'memorises' all data. This is to be used for evoluationary
    algorithms.
    -- kirthevasank
"""

import logging
import numpy as np
# local
from cilantro.learners.base_learner import LearningModel


logger = logging.getLogger(__name__)


class MemorisingModel(LearningModel):
    """ A learner which simply memorises all data. """

    def __init__(self, name, int_lb=-np.inf, int_ub=np.inf):
        """ Constructor. """
        super().__init__(name, int_lb, int_ub)
        self.data_x_vals = []
        self.data_y_vals = []
        self.data_sigmas = []

    # Add/process data -----------------------------------------------------------------------------
    def update_model_with_new_data(self, X, Y, Sigmas):
        """ Updates model with the given data. """
        self.data_x_vals.extend(X)
        self.data_y_vals.extend(Y)
        self.data_sigmas.extend(Sigmas)

    def _initialise_model_child(self):
        """ Initialise model. """
        pass

    # Recommendations ------------------------------------------------------------------------------
    def get_recommendation(self, perf_goal, *args, **kwargs):
        """ Obtain recommendation. """
        raise ValueError('No need to call this method.')

    def get_recommendation_for_upper_bound(self, perf_goal, *args, **kwargs):
        """ Obtain recommendation. """
        raise ValueError('No need to call this method.')

    def get_recommendation_for_lower_bound(self, perf_goal, *args, **kwargs):
        """ Obtain recommendation. """
        raise ValueError('No need to call this method.')

    def compute_conf_interval_for_input(self, x):
        """ Compute a confidence interval for the input point x. """
        raise ValueError('No need to call this method.')

    def compute_estimate_for_input(self, x):
        """ Compute estimate for the input point x. """
        raise ValueError('No need to call this method.')

