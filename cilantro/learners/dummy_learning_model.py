"""
    A Dummy learning model.
    -- kirthevasank
"""

# Local
from cilantro.learners.base_learner import LearningModel


class DummyLearningModel(LearningModel):
    """ A dummy learning model. """

    # Add/process data -----------------------------------------------------------------------------
    def update_model_with_new_data(self, X, Y, Sigmas):
        """ Updates model with the given data. """
        pass

    # Recommendations ------------------------------------------------------------------------------
    def get_recommendation(self, perf_goal, *args, **kwargs):
        """ Obtain recommendation. """
        return 0.5

    def get_recommendation_for_upper_bound(self, perf_goal, *args, **kwargs):
        """ Obtain recommendation. """
        return 1.0

    def get_recommendation_for_lower_bound(self, perf_goal, *args, **kwargs):
        """ Obtain recommendation. """
        return 0.0

    def compute_conf_interval_for_input(self, x):
        """ Compute a confidence interval for the input point x. """
        return (0.0, 1.0)

