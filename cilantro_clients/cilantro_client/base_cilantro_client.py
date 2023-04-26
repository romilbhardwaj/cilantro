import argparse
import logging
import time
from typing import Union, List

from cilantro_clients.alloc_sources.base_alloc_source import BaseAllocSource
from cilantro_clients.data_sources.base_data_source import BaseDataSource
from cilantro_clients.publishers.base_publisher import BasePublisher, CLIENT_RETCODE_SUCCESS

logger = logging.getLogger(__name__)


class BaseCilantroClient(object):
    def __init__(self,
                 data_source: BaseDataSource,
                 publisher: Union[BasePublisher, List[BasePublisher]],
                 poll_frequency: float,
                 alloc_source: BaseAllocSource = None,
                 ):
        '''
        Base cilantro client that polls a data_source and sends the data to the publisher.
        :param data_source: DataSource object to poll for data
        :param publisher: Publisher object to send data to or a list of publisher objects.
        :param poll_frequency: How frequently to poll data from data_source
        :param alloc_source: Allocation source to use to fetch current allocation if 'alloc' is not returned by data source.
        '''
        self.data_source = data_source
        if not isinstance(publisher, list):
            publisher = [publisher]
        self.publishers = publisher
        self.poll_frequency = poll_frequency
        self.alloc_source = alloc_source

    def run_loop(self):
        while True:
            try:
                data = self.data_source.get_data()
            except StopIteration as e:
                logger.info(f"{str(e)}")
                break
            if data is not None:
                if 'alloc' not in data and self.alloc_source is not None:
                    data['alloc'] = self.alloc_source.get_allocation()
                for p in self.publishers:
                    ret, msg = p.publish(data)
                    if ret != CLIENT_RETCODE_SUCCESS:
                        logger.warning(f"Publishing to {type(p)} failed, not retrying. Error: {msg}")
            time.sleep(self.poll_frequency)

    @classmethod
    def add_args_to_parser(self,
                           parser: argparse.ArgumentParser):
        parser.add_argument('--poll-frequency', '-pf', type=float, default=1,
                            help='How frequently to poll the data source.')
