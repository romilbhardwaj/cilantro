"""
    A Simple data logger using pandas.
    -- kirthevasank
"""
import json
import logging
import time
from typing import List, Dict

import pandas as pd
import numpy as np
# Local
from cilantro.data_loggers.base_data_logger import BaseDataLogger


logger = logging.getLogger(__name__)


class SimpleDataLogger(BaseDataLogger):
    """ A data logger backed by Pandas """
    DEBUG_FIELD_PREFIX = 'DEBUG.'
    def __init__(self,
                 app_id: str,
                 fields: List[str],
                 index_fld: str,
                 max_inmem_table_size=-1,
                 workload_type=None):
        """
        Constructor.
        :param app_id: A tag for the datalogger. Useful for referencing.
        :param fields: a list of fields that we will store in our table. If a
          field name is prepended with DEBUG., then the debug field is json
          deserialized to extract the field. E.g., DEBUG.myfield will json load
          the debug field in the utilitymessage and get the myfield value.
          Use the regular name (e.g. myfield) to access these debug fields when
          needed.
        :param index_fld: Field to be used as the index
        :max_inmem_table_size: is the maximum size to be stored in memory.
        :disk_file, disk_dir: location to store in file.
        :write_to_disk_every: time (in seconds) at which we should write to file.
        """
        self.app_id = app_id
        self._raw_fields = fields
        self._df_fields = [f.replace(self.DEBUG_FIELD_PREFIX, '') for f in self._raw_fields]
        self._index_fld = index_fld
        self._table = pd.DataFrame(columns=self._df_fields)
        self._num_data_logged = 0
        # For writing to file
        self._max_inmem_table_size = max_inmem_table_size
        self._last_write_to_disk_timestamp = -np.inf
        self._writing_loop_running = False
        self._poll_time = 0.5
        self.file_lock = None
        self._disk_file_name = self.app_id + '_' + workload_type + '.csv' if workload_type else self.app_id + '.csv'
        self._write_to_file_path = None
        # Call super constructor -----------------------------------
        super().__init__()

    def get_disk_file_name(self):
        """ Returns disk file name. """
        return self._disk_file_name

    def write_to_disk_routine(self, write_to_file_path):
        """ This actually writes to disk. """
        if self._write_to_file_path:
            assert self._write_to_file_path == write_to_file_path
        else:
            self._write_to_file_path = write_to_file_path
        write_indices = sorted([elem for elem in self._table.index if
                                self._last_write_to_disk_timestamp < elem])
        if len(write_indices) > 0:
            sub_df = self._table.loc[write_indices]
            if sub_df.shape[0] > 0:
                use_header_when_w2d = not np.isfinite(self._last_write_to_disk_timestamp)
                to_csv_mode = 'w' if use_header_when_w2d else 'a'
                sub_df.to_csv(self._write_to_file_path,
                              header=use_header_when_w2d, mode=to_csv_mode)
                self._last_write_to_disk_timestamp = write_indices[-1]
                logger.debug('[leaf=%s] Wrote %d data to %s.', self.app_id, sub_df.shape[0],
                             self._write_to_file_path)
        # Check if we have exceeded the in-memory size -------------------------------------------
        if self._table.shape[0] > self._max_inmem_table_size:
            self._table = self._table.tail(self._max_inmem_table_size)

    def log_event(self, event: Dict):
        """ Logs event into the pandas table """
        new_data = []
        debug_json = None
        try:
            debug_json = json.loads(event['debug'])
        except:
            # No worries if json deserialization failed, maybe it's not required
            pass
        for fld in self._raw_fields:
            if fld.startswith(self.DEBUG_FIELD_PREFIX):
                # Need to extract from debug json
                if debug_json is None:
                    raise ValueError(f"Tried to extract {fld} from debug field "
                                     f"in event, but json deserialization "
                                     f"failed. Debug was: {event['debug']}")
                else:
                    val = debug_json[fld.replace(self.DEBUG_FIELD_PREFIX, '')]
            else:
                val = event[fld]
            new_data.append(val)
        self._table.loc[event[self._index_fld]] = new_data
        self._num_data_logged += 1

    def get_data(self,
                 fields: List[str],
                 start_time_stamp: float = None,
                 end_time_stamp: float = None):
        """ Gets data between the given time range """
        # pylint: disable=broad-except
        # pylint: disable=logging-not-lazy
        all_fields = fields
        start_time_stamp = start_time_stamp if start_time_stamp else 0
        end_time_stamp = end_time_stamp if end_time_stamp else np.inf
        ret_indices = sorted([elem for elem in self._table.index if
                              start_time_stamp < elem <= end_time_stamp])
        ret = []
        for idx in ret_indices:
            curr_data = {fld: self._table.loc[idx][fld] for fld in all_fields}
            ret.append(curr_data)
        last_time_stamp = ret_indices[-1] if ret_indices else start_time_stamp
        # First check if we need to load from file -------------------------------------
        # Below, the first condition checks if we are writing to file at all, the second if
        # the requested time stamp is earlier than the last written time stamp, and the third
        # if there is any data on file that is not in memory.
        if (self._write_to_file_path is not None) and \
           (start_time_stamp < self._last_write_to_disk_timestamp) and \
           (self._table.shape[0] < self._num_data_logged):
            while self.file_lock: # wait for the lock to be released
                time.sleep(self._poll_time/2)
            self.file_lock = 'get_data'
            try:
                # pylint: disable=no-member
                df = pd.read_csv(self._write_to_file_path, index_col=0)
                latest_file_time_stamp = ret_indices[0]
                file_ret_indices = sorted([elem for elem in df.index if
                                           start_time_stamp < elem <= latest_file_time_stamp])
                file_ret = []
                for file_ret_idx in file_ret_indices:
                    curr_file_data = {fld: df.loc[file_ret_idx][fld] for fld in all_fields}
                    file_ret.append(curr_file_data)
                ret = file_ret + ret
#                 logger.info(('[leaf=%s:] Read %d from %s, but added only %d. start_ts=%0.4f, ' +
#                              'latest_ts=%0.4f, end_ts=%0.4f, file_idxs=[%0.4f, %0.4f]'),
#                             self.app_id, df.shape[0], self._disk_file, len(file_ret_indices),
#                             start_time_stamp, latest_file_time_stamp, end_time_stamp,
#                             min(df.index), max(df.index))
            except Exception as e:
                logger.info('[leaf=%s:] Failed file read with %s.', self.app_id, str(e))
            self.file_lock = None
        # ------------------------------------------------------------------------------
#         logger.debug(('[leaf=%s:] Returning %d data on get_data. num_data_logged=%d, ' +
#                       'in_mem_df_size=%d.'), self.app_id, len(ret), self._num_data_logged,
#                        self._table.shape[0])
        return ret, last_time_stamp

