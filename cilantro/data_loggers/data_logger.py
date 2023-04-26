"""
    Data logger.
    DEPRECATED - Use simple_data_logger.py instead --kirthevasank
"""

from __future__ import annotations
from datetime import time
from io import StringIO
from numpy.lib.shape_base import split
from pandas import DataFrame as df
from pandas import read_csv
from pandas import concat
import numpy as np
from statsmodels.tsa.ar_model import AutoReg
from typing import TypeVar, Generic
import os

T = TypeVar('T')

class DataLogger(object):
    def __init__(self, max_local_data_table_size: int, disk_file: str):
        self.current_data = None
        self.earliest_timestamp = 0
        self.latest_timestamp = 0
        self.max_local_data_table_size = max_local_data_table_size
        self.filepath = disk_file
    
    def split_table(self, split_size: int=2):
        # store first 1/split_size of current data
        split_tables = np.array_split(self.current_data, split_size)
        new_current_data = None
        for table in split_tables[1:]:
            if new_current_data is None:
                new_current_data = table
            else:
                new_current_data = np.concatenate(new_current_data, table)
        self.current_data = new_current_data

        values_to_store = split_tables[0]
        try:
            old_values = read_csv(self.filepath, header=0, index_col=0, encoding='utf-8')
            old_values = old_values.append(values_to_store, ignore_index=True)
        except:
            old_values = values_to_store
        old_values.to_csv(self.filepath)

        # update earliest timestamp
        self.earliest_timestamp = self.current_data['timestamp'].iloc[0]

    def log_data(self, timestamp: float, data: dict[str, T]):
        '''
        Logs an event
        :param event:
        :return: None
        '''
        data['timestamp'] = timestamp
        new_data = df.from_dict(data, orient="columns")

        cols = new_data.columns.tolist()
        cols = cols[-1:] + cols[:-1]
        new_data = new_data[cols]
        if (self.current_data is None):
            self.earliest_timestamp = timestamp
            self.latest_timestamp = timestamp
            self.current_data = new_data
        else:
            self.current_data = self.current_data.append(new_data, ignore_index=True)
            self.latest_timestamp = timestamp
            if (len(self.current_data.index) > self.max_local_data_table_size):
                self.split_table()
        
    def get_data(self, start_timestamp, end_timestamp):
        current_filtered_values = self.current_data[(self.current_data['timestamp'] >= start_timestamp) & (self.current_data['timestamp'] <= end_timestamp)]
        if start_timestamp < self.earliest_timestamp:
            old_values = read_csv(self.filepath, header=0, index_col=0)
            filtered_values = old_values[(old_values['timestamp'] >= start_timestamp) & (old_values['timestamp'] <= end_timestamp)]
            current_filtered_values = filtered_values.append(current_filtered_values, ignore_index=True)
        return current_filtered_values # returns dataframe object w filtered data between timestamps

    def get_prediction(self, target_timestamp):
        data = self.current_data.data
        model_fit = AutoReg(data, lags=1).fit()
        return model_fit.predict(start=target_timestamp, end=target_timestamp+1)
