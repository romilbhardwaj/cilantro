"""
Base class for a log parser.
"""
from argparse import ArgumentParser
from typing import Dict


class BaseLogParser(object):
    def __init__(self):
        pass

    def get_data(self, log_file) -> Dict:
        """
        This method returns a dict with data by parsing the log_file.
        The dict must contain 'load', 'reward', 'sigma' keys.
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