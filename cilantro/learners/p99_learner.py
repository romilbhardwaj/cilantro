"""
    A learner for the p99 latency.
    -- kirthevasank
"""

import logging
# local
from cilantro.learners.base_learner import BaseLearner

# How the learner works: The learner polls the data logger for new data. If it receives new
# data, it processes the data and updates the model and keeps repeating this infinitely.

DFLT_SLEEP_TIME_BETWEEN_DATA_REPOLLS = 10
NUM_INIT_DATA_TO_IGNORE = 5

logger = logging.getLogger(__name__)


class P99Learner(BaseLearner):
    """ P99 Learner. """

    def _fetch_and_format_latest_data(self):
        """ Fetches and formats the latest data.
            Overrides BaseLearner to set the Reward to the negative p99 latency, Allocs to a
            dictionary of allocations for each leaf.
        """
        new_data, new_time_stamp = \
            self.data_logger.get_data(fields=['load', 'p99', 'allocs',
                                              'event_start_time', 'event_end_time'],
                                      start_time_stamp=self.current_time_stamp,
                                      end_time_stamp=None)
        Allocs = [elem['allocs'] for elem in new_data]
        Rewards = [-elem['p99'] for elem in new_data]
        Loads = [elem['load'] for elem in new_data]
        Sigmas = None
        Event_times = [elem['event_end_time'] - elem['event_start_time'] for elem in new_data]
        return Allocs, Rewards, Loads, Sigmas, Event_times, new_time_stamp

    def get_mean_pred_and_std_for_alloc_load(self, *args, **kwargs):
        """ Returns mean prediction and std from the model. """
        model_for_serving = self.get_model_for_serving()
        return model_for_serving.get_mean_pred_and_std_for_alloc_load(*args, **kwargs)

