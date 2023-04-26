"""
    A bank for learners.
    -- kirthevasank
    -- romilbhardwaj
"""

import logging
import random
import threading
import time
from cilantro.core.bank import Bank


logger = logging.getLogger(__name__)


class DemandRecommender:
    """ An abstraction for recommending demands. """

    def initialise(self):
        """ Initialise. """
        raise NotImplementedError('Implement in a child class.')

    def get_recommendation(self, perf_goal, load, *args, **kwargs):
        """ Obtain recommendation. """
        raise NotImplementedError('Implement in a child class.')

    def get_recommendation_for_upper_bound(self, perf_goal, load, *args, **kwargs):
        """ Obtain recommendation. """
        raise NotImplementedError('Implement in a child class.')

    def get_recommendation_for_lower_bound(self, perf_goal, load, *args, **kwargs):
        """ Obtain recommendation. """
        raise NotImplementedError('Implement in a child class.')


class BaseLearnerBank(Bank):
    """ Base learner bank. """

    def __init__(self, num_parallel_training_threads=None, sleep_time_between_trains=None):
        """ Constructor. """
        super().__init__()
        self.num_parallel_training_threads = num_parallel_training_threads
        self.sleep_time_between_trains = sleep_time_between_trains
        self._training_thread_running = False

    def initiate_training_loop(self):
        """ Initiate training loop. """
        if self.num_parallel_training_threads is None:
            raise ValueError('num_parallel_training_threads not set!')
        if not self._training_thread_running:
            self._training_thread_running = True
            thread = threading.Thread(target=self._training_loop)
            thread.start()

    def stop_training_loop(self):
        """ Stops the training loop. """
        self._training_thread_running = False

    def _training_loop(self):
        """ Training loop. """
        while self._training_thread_running:
            all_tags = self.get_tags()
            num_tags = len(all_tags)
            if num_tags == 0:
                time.sleep(10 * self.sleep_time_between_trains)
                continue
            # Sleep for some time --------------------------------------------------
            time.sleep(self.sleep_time_between_trains)
            # First assign each leaf to different threads --------------------------
            random.shuffle(all_tags)
            parallel_bins = [[] for _ in range(min(num_tags, self.num_parallel_training_threads))]
            num_parallel_bins = len(parallel_bins)
            bin_counter = 0
            for tag in all_tags:
                parallel_bins[bin_counter].append(tag)
                bin_counter = (bin_counter + 1) % num_parallel_bins
            # Create threads for the first num_parallel_bins - 1 bins --------------
            threads = [threading.Thread(target=self._training_routine_for_given_tags, args=(pbin, ))
                       for pbin in parallel_bins[:-1]] # last bin will be done by current thread
            for thr in threads:
                thr.start()
            # Process the last bin in current thread -------------------------------
            self._training_routine_for_given_tags(parallel_bins[-1])
            # Join threads ---------------------------------------------------------
            for thr in threads:
                thr.join()

    def _training_routine_for_given_tags(self, tags):
        """ Training routine. """
        for tag in tags:
            self.get(tag).model_update_routine()


class LearnerBank(BaseLearnerBank):
    """ Learner Bank. """

    @classmethod
    def _check_type(cls, obj):
        """ Checks type. """
        assert isinstance(obj, DemandRecommender)

