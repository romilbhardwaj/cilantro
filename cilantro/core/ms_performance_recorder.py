"""
    A peformance recorder class for the microservices use case.
    -- romilbhardwaj
    -- kirthevasank
"""

import logging
import pickle
import threading
import time
# Local
from cilantro.core.bank import Bank

logger = logging.getLogger(__name__)


NUM_DATA_FOR_RECENT_REPORTS = 8


class MSPerformanceRecorder:
    """ Performance recorder for microservices. """

    def __init__(self, descr, data_logger, fields_to_report):
        """ Constructor. """
        super().__init__()
        self.descr = descr
        self.data_logger = data_logger
        self.fields_to_report = fields_to_report
        self.history = []
        self.history_results = {}
        # For computing and tracking the utility -------------------------------------------------
        self._curr_time_stamp = None
        self._curr_total_vals = {fld: 0 for fld in self.fields_to_report}
        self._curr_mean_vals = {fld: 0 for fld in self.fields_to_report}
        self._recent_mean_vals = {fld: 0 for fld in self.fields_to_report}
        self._event_times = []
        self._curr_total_time = 0

    @classmethod
    def _compute_vals_with_data_batch(cls, data_batch, fields_to_report):
        """ Computes values with the data batch. """
        batch_time = 0
        batch_tot_vals = {fld: 0 for fld in fields_to_report}
        for elem in data_batch:
            curr_time = elem['event_end_time'] - elem['event_start_time']
            batch_time += curr_time
            for fld in fields_to_report:
                batch_tot_vals[fld] += elem[fld] * curr_time
        # compute the mean values
        batch_mean_vals = {fld: tot_val/batch_time for fld, tot_val in batch_tot_vals.items()}
        return batch_mean_vals

    def _update_vals_with_new_data(self, new_data):
        """ updates history. """
        for elem in new_data:
            curr_time = elem['event_end_time'] - elem['event_start_time']
            self._event_times.append(curr_time)
            self._curr_total_time += curr_time
            for fld in self.fields_to_report:
                self._curr_total_vals[fld] += elem[fld] * curr_time
        # Compute the mean values ----------------------------------------------------
        self._curr_mean_vals = {fld: tot_val/self._curr_total_time for fld, tot_val in
                                self._curr_total_vals.items()}

    def fetch_data_and_update_history(self):
        """ Fetch data and update history. """
        new_data, new_time_stamp = self.data_logger.get_data(
            fields=self.fields_to_report + ['event_end_time', 'event_start_time'],
            start_time_stamp=self._curr_time_stamp, end_time_stamp=None)
        logger.debug('Received %d new data in %s.', len(new_data), self.descr)
        if new_data:
            self.history.extend(new_data)
            self._update_vals_with_new_data(new_data)
            self._recent_mean_vals = self._compute_vals_with_data_batch(
                self.history[-NUM_DATA_FOR_RECENT_REPORTS:], self.fields_to_report)
            self._curr_time_stamp = new_time_stamp
            self.history_results[self._curr_time_stamp] = \
                {'all': self._curr_mean_vals, 'recent': self._recent_mean_vals}
            self._most_recent_data = new_data
        else:
            self._most_recent_data = []
        return new_data

    def get_report_str(self):
        """ Obtains strings that can be used for reporting. """
        all_str = '(%0.1f):: '%(self._curr_total_time)
        recent_str = ''
        for fld in self.fields_to_report:
            all_str += '%s=%0.3f, '%(fld, self._curr_mean_vals[fld])
            recent_str += '%s=%0.3f, '%(fld, self._recent_mean_vals[fld])
        return all_str, recent_str

    def get_most_recent_data(self):
        """ Returns the most recent data. """
        return self._most_recent_data

    def get_all_data(self):
        """ Returns the most recent data. """
        return self.history


class MSPerformanceRecorderBank(Bank):
    """ Bank for MS Performance recorder. """

    def __init__(self, resource_quantity, alloc_granularity=None,
                 report_results_every=-1, save_file_name=None, report_results_descr=''):
        """ Constructor.
            - report_results_every is the time (in seconds) interval at which the results must be
              reported (and saved).
            - resource_quantity, alloc_granularity are the total number of resources and the
              allocation granularity.
            - save_file_name, if a string, is where the results need to be saved.
        """
        super().__init__()
        self.report_results_every = report_results_every
        self.resource_quantity = resource_quantity
        self.alloc_granularity = alloc_granularity
        self.save_file_name = save_file_name
        self.history = []
        if self.report_results_every > 0:
            self._report_results_poll_time = self.report_results_every / 10
            self._last_report_time = None
            self._reporting_loop_running = False
            self._report_results_descr = report_results_descr

    @classmethod
    def _check_type(cls, obj):
        """ Checks type. """
        assert isinstance(obj, MSPerformanceRecorder)

    def initiate_report_results_loop(self):
        """ Initiates the report results loop. """
        self._last_report_time = time.time()
        if not self._reporting_loop_running:
            self._reporting_loop_running = True
            thread = threading.Thread(target=self._report_results_loop, args=())
            thread.start()

    def _save_results(self):
        """ Saves results. """
        if self.save_file_name:
            curr_to_save = {key: {'history': val.history, 'history_results': val.history_results}
                            for key, val in self.enumerate()}
            with open(self.save_file_name, 'wb') as pickle_save_file:
                pickle.dump(curr_to_save, pickle_save_file)
                pickle_save_file.close()

    def _report_results_loop(self):
        """ Reports results in this loop. """
        # pylint: disable=broad-except
        while self._reporting_loop_running:
            time.sleep(self._report_results_poll_time)
            curr_time = time.time()
            if curr_time >= self._last_report_time + self.report_results_every:
                for key, mspb in self.enumerate():
                    mspb.fetch_data_and_update_history()
                    all_rep_str, recent_rep_str = mspb.get_report_str()
                    print_str = key + ' ' + all_rep_str + '.   Recent:' + recent_rep_str
                    logger.info(print_str)
#                     try:
#                         mspb.fetch_data_and_update_history()
#                         all_rep_str, recent_rep_str = mspb.get_report_str()
#                         print_str = key + ' ' + all_rep_str + '\n    - recent:' + recent_rep_str
#                         logger.info(print_str)
#                     except Exception as e:
#                         err_str = ('MSPerfBank (ts=%0.2f): Recent results could'nt be computed.' +
#                                    ' Exception: %s')%(curr_time, e)
#                         logger.info(err_str)
                self._last_report_time = curr_time

