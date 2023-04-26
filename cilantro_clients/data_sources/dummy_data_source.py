"""
A dummy data source which always returns a fixed dictionary.
"""
import time
from typing import Dict

from cilantro_clients.data_sources.base_data_source import BaseDataSource


class DummyDataSource(BaseDataSource):
    def __init__(self, default_data = None):
        if default_data is None:
            default_data = {
                'load': 1,
                'alloc': 1,
                'reward': 1,
                'sigma': 1,
                'event_start_time': 0,
                'event_end_time': 0
            }
        self.default_data = default_data
        self.last_get_time = 0
        super(BaseDataSource, self).__init__()

    def get_data(self) -> Dict:
        now = time.time()
        self.default_data['event_start_time'] = self.last_get_time
        self.default_data['event_end_time'] = now
        self.last_get_time = now
        return self.default_data
