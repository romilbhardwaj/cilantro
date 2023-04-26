# Maybe defunct with the new class structure.

"""
    A dummy learner.
    -- romilbhardwaj
    -- kirthevasank
"""

import logging
import time

from cilantro.learners.base_learner import LearningModel

logger = logging.getLogger(__name__)


class DummyLearner(LearningModel):
    """ Dummy learner. """

    def __init__(self,
                 app_id: str,
                 sleep_time: int = 5):
        """ Constructor. """
        self.sleep_time = sleep_time
        super(DummyLearner, self).__init__(app_id)

    def learn(self, train_data):
        time.sleep(self.sleep_time)
        logger.debug("Dummy learning done.")

    # TODO: What is the purpose of training_task?
    def set_training_task(self,t):
        self.training_task = t

    # Add/process data -----------------------------------------------------------------------------
    def add_data_point(self, x, y, sigma):
        """ Add a single data point. """
        time.sleep(self.sleep_time)
        logger.debug("Dummy learning done.")

    def compute_conf_interval_for_input(self, x):
        """ Compute a confidence interval for the input point x. """
        return (0, 1)

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

