"""
Fetches metrics from cassandra servers and returns them
"""
from typing import Dict

from cilantro_clients.data_sources.base_data_source import BaseDataSource


class CassandraDataSource(BaseDataSource):
    def __init__(self):
        raise NotImplementedError

    def get_data(self) -> Dict:
        raise NotImplementedError