"""
Base class for a data source for cilantro client.
"""
from argparse import ArgumentParser
from typing import Dict


class BaseDataSource(object):
    def __init__(self):
        pass

    def get_data(self) -> Dict:
        """
        This method returns a dict with data.
        :raises: raise StopIteration if you want the run_loop of the client to terminate.
        :return:
        """
        raise NotImplementedError('Implement in a child class.')

    @classmethod
    def add_args_to_parser(self,
                           parser: ArgumentParser):
        """
        Adds class specific arugments to the given argparser.
        Useful to quickly add args to driver script for different sources.
        :param parser: argparse object
        :return: None
        """
        pass