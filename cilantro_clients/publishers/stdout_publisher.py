"""
Publishes data to stdout.
"""
import logging
from typing import Dict

from cilantro_clients.publishers.base_publisher import BasePublisher, CLIENT_RETCODE_SUCCESS

logger = logging.getLogger(__name__)

class StdoutPublisher(BasePublisher):
    def __init__(self):
        """
        Prints the data object as is to stdout.
        """
        self.count = 0
        super(BasePublisher, self).__init__()

    def publish(self, data: Dict) -> [int, str]:
        """
        Publishes data to the output sink and returns a ret code.
        :param data: Dictionary of data to be published
        :return: retcode, 1 if successful.
        """
        print(f"STDOUT_Publisher[Msg {self.count}]: " + str(data), flush=True)
        self.count += 1
        return CLIENT_RETCODE_SUCCESS, None

