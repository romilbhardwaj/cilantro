"""
    A data source that provides methods to read log folders.
    -- romilbhardwaj
    -- kirthevasank
"""

import logging
import os
import time
from argparse import ArgumentParser
from typing import Dict
import glob
# Local imports
from cilantro_clients.data_sources.base_data_source import BaseDataSource
from cilantro_clients.data_sources.log_parsers.base_log_parser import BaseLogParser

logger = logging.getLogger(__name__)

class LogFolderDataSource(BaseDataSource):
    """
    This data source reads log files from a directory.
    It maintains a pointer to the last file read so that get_data is called only
    on the latest files written to the log folder by the workload client.
    """
    def __init__(self,
                 log_dir_path: str,
                 log_parser: BaseLogParser,
                 log_extension: str = '*.log'):
        self.log_dir_path = log_dir_path
        self.log_parser = log_parser
        self.log_extension = log_extension
        self.last_get_time = time.time()
        self.last_file_read = None
        super().__init__()

    def get_data(self) -> Dict:
        """
        The get_data for LogFolderDatasource aggregates updates since the last update by parsing all
        output files produced in the time period since the last get_data call.
        The returned dict must contain 'load', 'alloc', 'reward', 'sigma' keys.
        :return:
        """
        now = time.time()
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
        report_str = f"Got {len(files_to_analyze)} new files to analyze. " + \
                     f"Last file was {self.last_file_read}"
        logger.info(report_str)

        # Parse the log files and get the data dictionary for each file
        results = {}
        for f in files_to_analyze:
            try:
                latest_results = self.log_parser.get_data(f)
                if not (latest_results is None):
                    results[f] = latest_results
#                 results[f] = self.log_parser.get_data(f)
            except LookupError as e:
                report_str = f"Log parsing failed for {f}. Skipping file. Error was: {str(e)}"
                logger.error(report_str)


        # Aggregate results across files into one dictionary
        # Aggregate = mean of values across dictionaries.
#         KEYS_TO_AGG = ['load', 'reward']
        if results:
            # Get the latest data
            agg_result = results[sorted_files[0]]
#             # sample_keys = list(results.values())[0].keys()
#             for k in KEYS_TO_AGG:
#                 agg_result[k] = sum(d[k] for d in results.values()) / len(results)
#             # Add timestamps
#             agg_result['event_start_time'] = self.last_get_time
#             agg_result['event_end_time'] = now
            self.last_get_time = now
        else:
            agg_result = None

        if files_to_analyze:
            self.last_file_read = sorted_files[0]
        return agg_result

    @classmethod
    def add_args_to_parser(cls,
                           parser: ArgumentParser):
        parser.add_argument('--log-folder-path', '-lfp', type=str,
                            help='Path to log folder')
        parser.add_argument('--log-extension', '-lext', type=str,
                            default='*.log')

