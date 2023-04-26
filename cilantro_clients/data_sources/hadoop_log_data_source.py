"""
    A data source for the Hadoop workload.
    -- romilbhardwaj
    -- kirthevasank

    NB:
    - If you want to specify the dataset size for the file, you should do so via the
      hadoop_dataset_size argument for the HadoopLogFileParser object. Right now, we are computing
      the throughput as fraction-of-file-per-sec, but you could convert it into megabytes-per-sec or
      any other quantity via this argument.
    - min_perc_for_event and min_time_for_event are arguments which control the minimum time to wait
      before declaring a new event.
"""

from argparse import ArgumentParser
from datetime import datetime
import logging
import os
import re
import time
from typing import Dict
import glob

from cilantro_clients.data_sources.base_data_source import BaseDataSource

logger = logging.getLogger(__name__)

HADOOP_PATTERN_REGEX = r'^(\d{2})/(\d{2})/(\d{2}) (\d{2}):(\d{2}):(\d{2}) INFO Job:([ ]{2,})map'
# E.g: 21/08/27 20:33:48 INFO Job:  map 14% reduce 0%

class HadoopLogFileParser:
    """ Parses a given hadoop log file. """

    def __init__(self, file_name, hadoop_dataset_size=1, min_perc_for_event=5,
                 min_time_for_event=10):
        """ Constructor.
            min_perc_for_event is the minimum percentage to wait to create an event.
        """
        super().__init__()
        self.file_name = file_name
        self.last_read_line_num = 0
        self.hadoop_dataset_size = hadoop_dataset_size
        self.min_perc_for_event = min_perc_for_event
        self.min_time_for_event = min_time_for_event
        self._data_being_processed = []
        self._last_time_read = None
        self._last_perc_read = None
        self._reduce_started = False

    def _process_curr_line(self, line):
        """ processes current line. """
        if self._reduce_started:
            return
        line_elems = line.split()
        # Process the date --------------------
        curr_date = line_elems[0].split('/')
        curr_year = int('20' + curr_date[0])
        curr_month = int(curr_date[1])
        curr_day = int(curr_date[2])
        # process the time --------------------
        curr_time = line_elems[1].split(':')
        curr_hour = int(curr_time[0])
        curr_min = int(curr_time[1])
        curr_sec = int(curr_time[2])
        # Create a time stamp
        curr_time_read = datetime(curr_year, curr_month, curr_day, curr_hour, curr_min, curr_sec)
        curr_perc_read = float(line_elems[5][:-1]) # 5th elem is the percentage, last literal is %
        reduce_perc_read = float(line_elems[7][:-1]) # 5th elem is the percentage, last literal is %
        if reduce_perc_read > 0:
            self._reduce_started = True
            return
        if self._last_perc_read is not None:
            time_gap = (curr_time_read - self._last_time_read).total_seconds()
            perc_gap = curr_perc_read - self._last_perc_read
            if time_gap > 0 and perc_gap > 0:
                self._data_being_processed.append({
                    'sub_start_time': self._last_time_read,
                    'sub_end_time': curr_time_read,
                    'sub_perc_complete': perc_gap,
                    'tot_perc_complete': curr_perc_read,
                    'sub_time_gap': time_gap})
                self._last_perc_read = curr_perc_read
                self._last_time_read = curr_time_read
        else:
            self._last_perc_read = curr_perc_read
            self._last_time_read = curr_time_read

    def _get_next_data_elem(self):
        """ Returns the next data list. """
        if len(self._data_being_processed) == 0:
            return None
        # Return a dictionary ---------------------------------------------------------
        curr_sub_event = self._data_being_processed.pop(0)
        ret = {}
        ret['load'] = 1.0
        ret['event_start_time'] = curr_sub_event['sub_start_time'].timestamp()
        curr_tot_time = curr_sub_event['sub_time_gap']
        curr_tot_perc = curr_sub_event['sub_perc_complete']
        while ((curr_tot_perc < self.min_perc_for_event) or
                (curr_tot_time < self.min_time_for_event)) \
            and (len(self._data_being_processed) > 0):
            curr_sub_event = self._data_being_processed.pop(0)
            curr_tot_time += curr_sub_event['sub_time_gap']
            curr_tot_perc += curr_sub_event['sub_perc_complete']
        ret['curr_completed_frac'] = self.hadoop_dataset_size * (curr_tot_perc/100)
        ret['tot_completed_frac'] = curr_sub_event['tot_perc_complete']
        ret['time_interval'] = curr_tot_time
        ret['reward'] = ret['curr_completed_frac']/ret['time_interval']
        ret['sigma'] = self.hadoop_dataset_size * 0.5 / ret['time_interval']
        ret['debug'] = 'hadoop test debug string'
        ret['event_end_time'] = curr_sub_event['sub_end_time'].timestamp()
        return ret

    def _get_current_data_list(self):
        """ Returns the current data. """
        ret = []
        while True:
            next_elem = self._get_next_data_elem()
            if next_elem is None:
                break
            ret.append(next_elem)
        print_str = 'HadoopLogFileParser returning: \n --- ' + '\n ---'.join([str(elem) for
                                                                              elem in ret])
        logger.debug(print_str)
        return ret

    def get_data(self):
        """ Return data. """
        with open(self.file_name, 'r') as file_handle:
            all_lines = [elem.strip() for elem in file_handle.readlines()]
            file_handle.close()
        curr_lines = all_lines[self.last_read_line_num:]
        self.last_read_line_num = len(all_lines)
        # Now process the data --------------------------------------------------
        for line in curr_lines:
            match = re.match(HADOOP_PATTERN_REGEX, line)
            if match:
                self._process_curr_line(line)
        # Finally return the data list -----------------------------------------
        return self._get_current_data_list()

    def file_is_complete(self):
        """ Returns true if file is complete. """
        return self._reduce_started or  \
               ((self._last_perc_read is not None) and (self._last_perc_read >= 100) and
                (len(self._data_being_processed) == 0))


class HadoopLogDataSource(BaseDataSource):
    """
    Reads and parses log files for the Hadoop workload.
    """

    def __init__(self,
                 log_dir_path: str,
#                  log_parser: BaseLogParser,
                 log_extension: str = '*.log'):
        self.log_dir_path = log_dir_path
#         self.log_parser = log_parser
        self.log_extension = log_extension
        self.last_get_time = time.time()
        self.last_file_read = None
        super().__init__()
        # attributes for keeping track of which file is being read
        self.curr_file_being_read = None
        self.num_lines_read_in_curr_file_being_read = None
        self.log_parser_dict = {}

    def get_data(self) -> Dict:
        """
        The get_data for LogFolderDatasource aggregates updates since the last update by parsing all
        output files produced in the time period since the last get_data call.
        The returned dict must contain 'load', 'alloc', 'reward', 'sigma' keys.
        :return:
        """
        # Get the files in the log dir
        list_of_logs = glob.glob(os.path.join(self.log_dir_path, self.log_extension))
        # 1st element is the latest file
        sorted_files = sorted(list_of_logs, key=os.path.getctime, reverse=True)
        # Get the list of files to analyze in this call to get_data (discard older files)
        if self.last_file_read is None:
            files_to_analyze = sorted_files
        else:
            last_read_index = sorted_files.index(self.last_file_read)
            files_to_analyze = sorted_files[0:last_read_index]
        logger.info(f"Got {len(files_to_analyze)} new files to analyze. " +
                    f"Last file was {self.last_file_read}")
        # Create a log parser for each new file ---------------------------------------
        for fta in files_to_analyze:
            self.log_parser_dict[fta] = HadoopLogFileParser(fta)
        # Obtain the newest data from each log file parser ----------------------------
        ret = []
        for _, lfp in self.log_parser_dict.items():
            ret.append(lfp.get_data())
        if files_to_analyze:
            self.last_file_read = sorted_files[0]
        # Delete unused files from log_parser_dict -------------------------------------
        keys_to_delete_from_dict = []
        for key, parser in self.log_parser_dict.items():
            if parser.file_is_complete():
                keys_to_delete_from_dict.append(key)
        for key in keys_to_delete_from_dict:
            self.log_parser_dict.pop(key)
        return ret

    @classmethod
    def add_args_to_parser(cls, parser: ArgumentParser):
        parser.add_argument('--log-folder-path', '-lfp', type=str,
                            help='Path to log folder')
        parser.add_argument('--log-extension', '-lext', type=str,
                            default='*.log')

