"""
Base class for a data publisher for cilantro client.
"""
from argparse import ArgumentParser
from typing import Dict

CLIENT_RETCODE_SUCCESS = 1
CLIENT_RETCODE_FAIL = 2


class BasePublisher(object):
    def __init__(self):
        pass

    def publish(self, data: Dict) -> [int, str]:
        """
        Publishes data to the output sink and returns a ret code.
        :param data: Dictionary of data to be published
        :return: retcode, 1 if successful and a str error message, if unsuccesful.
        """
        raise NotImplementedError('Implement in a child class.')

    @classmethod
    def add_args_to_parser(self,
                           parser: ArgumentParser):
        """
        Adds class specific arugments to the given argparser.
        Useful to quickly add args to driver scripts.
        :param parser: argparse object
        :return: None
        """
        pass
