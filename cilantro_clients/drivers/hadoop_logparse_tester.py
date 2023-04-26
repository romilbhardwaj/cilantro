'''
    Tester for parsing the log file from the Hadoop workload. This first creates a thread to
    write logs (from a test file) and then calls the parse to parse the files being written.
    You can run this file from the experiments/testing/hadoop_log_parsing directory as follows.
        python ../../../cilantro_clients/drivers/hadoop_logparse_tester.py --log-folder-path \
            ./logs --test-log-file eg_hadoop_log
    -- kirthevasank
'''

import argparse
from datetime import datetime
import logging
import os
import shutil
import threading
import time
import numpy as np
# Cilantro imports
from cilantro_clients.cilantro_client.base_cilantro_client import BaseCilantroClient
from cilantro_clients.data_sources.hadoop_log_data_source import HadoopLogDataSource
from cilantro_clients.publishers.grpc_publisher import GRPCPublisher
from cilantro_clients.publishers.stdout_publisher import StdoutPublisher


class LogLoopWriter:
    """ Writes the log file in a loop. """

    def __init__(self, test_log_file, log_folder_path, max_num_lines_per_write=20,
                 max_sleep_time_between_writes=2):
        """ Constructor. """
        self.log_folder_path = log_folder_path
        self.max_num_lines_per_write = max(2, max_num_lines_per_write)
        self.max_sleep_time_between_writes = max_sleep_time_between_writes
        # Delete the log file directory ------------------------------------------------------
        if os.path.exists(self.log_folder_path):
            shutil.rmtree(self.log_folder_path)
        # Read test file ---------------------------------------------------------------------
        with open(test_log_file, 'r') as file_handle:
            write_lines = file_handle.readlines()
            self.write_lines = [elem.strip() for elem in write_lines]
            file_handle.close()
        # Create the log folder path ---------------------------------------------------------
        os.makedirs(self.log_folder_path)

    def write_to_log_dir(self):
        """ Writes to log dir. """
        while True:
            # In this loop, we will write to a new file -----------------------------
            curr_file_name = os.path.join(
                self.log_folder_path, datetime.now().strftime('%m%d%H%M%S') + '.log')
            curr_write_lines = self.write_lines[:]
            curr_write_prefix = ''
            while len(curr_write_lines) > 0:
                # 1. First, sleep for some time -------------------------------------
                curr_sleep_time = (0.5 + 0.5 * np.random.random()) * \
                                  self.max_sleep_time_between_writes
                time.sleep(curr_sleep_time)
                # 2. Write to the file ----------------------------------------------
                num_lines_to_write = min(2 + np.random.randint(self.max_num_lines_per_write-1),
                                         len(curr_write_lines))
                lines_to_write = curr_write_lines[:num_lines_to_write]
                curr_write_lines = curr_write_lines[num_lines_to_write:]
#                 print('Writing %d lines to %s'%(num_lines_to_write, curr_file_name))
                write_str = curr_write_prefix + '\n'.join(lines_to_write)
                with open(curr_file_name, 'a') as curr_file_handle:
                    curr_file_handle.write(write_str)
                    curr_file_handle.close()
                curr_write_prefix = '\n'
                # Here, we will write within a file ---------------------------------
            curr_file_handle.close()
            time.sleep(3 * self.max_sleep_time_between_writes)


def main():
    """ Main function. """
    logging.basicConfig(level=logging.DEBUG)

    parser = argparse.ArgumentParser(description='Hadoop log parser.')
    # Add parser args
    HadoopLogDataSource.add_args_to_parser(parser)
    GRPCPublisher.add_args_to_parser(parser)
    BaseCilantroClient.add_args_to_parser(parser)
    # Add args for this test
    parser.add_argument('--test-log-file', '-tlf', type=str,
                        help='The test log file to keep repeatedly writing.')
    args = parser.parse_args()
    print(args)

    # Start writing ----------------------------------------------------------
    log_loop_writer = LogLoopWriter(args.test_log_file, args.log_folder_path)
    thr = threading.Thread(target=log_loop_writer.write_to_log_dir, args=())
    thr.start()

    # Define objects:
    data_source = HadoopLogDataSource(log_dir_path=args.log_folder_path,
                                      log_extension=args.log_extension)
    publisher = StdoutPublisher()
    client = BaseCilantroClient(data_source,
                                publisher,
                                poll_frequency=args.poll_frequency)
    client.run_loop()

if __name__ == '__main__':
    main()

