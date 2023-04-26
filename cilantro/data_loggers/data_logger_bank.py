"""
    Bank to store data loggers.
    --kirthevasank
"""

import logging
import os
import threading
import time
# Local
from cilantro.core.bank import Bank


logger = logging.getLogger(__name__)


class DataLoggerBank(Bank):
    """ Simple data logger bank. """

    def __init__(self, write_to_disk_every=10, write_to_disk_dir=None):
        """ Constructor. """
        super().__init__()
        self._writing_loop_running = False
        self._write_to_disk_every = write_to_disk_every
        self._writing_loop_poll_time = write_to_disk_every / 10
        self._max_num_retries_for_lock_release = 10
        self.write_to_disk_dir = write_to_disk_dir
        if self.write_to_disk_dir and not os.path.exists(write_to_disk_dir):
            os.makedirs(write_to_disk_dir)

    def initiate_write_to_disk_loop(self):
        """ Initiate the write to disk loop. """
        self._last_write_to_disk_time = time.time()
        if not self.write_to_disk_dir:
            raise ValueError('write_to_disk_dir not provided in constructor!')
        self._last_write_to_disk_time = time.time()
        if not self._writing_loop_running:
            self._writing_loop_running = True
            thread = threading.Thread(target=self._write_to_disk_loop, args=())
            logger.info('Initiated writing to disk loop for data logger in dir %s.',
                        self.write_to_disk_dir)
            thread.start()

    def stop_write_to_disk_loop(self):
        """ Stops the report results loop. """
        self._writing_loop_running = False

    def _write_to_disk_loop(self):
        """ This loop writes to disk in a separate thread. """
        while self._writing_loop_running:
            time.sleep(self._writing_loop_poll_time)
            curr_time = time.time()
            if (curr_time >= self._last_write_to_disk_time + self._write_to_disk_every):
                self._last_write_to_disk_time = curr_time
                for tag, data_logger in self.enumerate(): # iterate through each data logger
                    write_to_file = os.path.join(self.write_to_disk_dir,
                                                 data_logger.get_disk_file_name())
                    lock_not_available_counter = 0
                    # wait for the lock to be released
                    while (not (data_logger.file_lock is None)) and \
                        lock_not_available_counter <= self._max_num_retries_for_lock_release:
                        lock_not_available_counter += 1
                        time.sleep(self._writing_loop_poll_time)
                    if lock_not_available_counter > self._max_num_retries_for_lock_release:
                        logger.info('Skipping writing data logger for %s as lock was not released.',
                                    tag)
                        continue
                    data_logger.file_lock = 'writing'
                    data_logger.write_to_disk_routine(write_to_file)
                    data_logger.file_lock = None

