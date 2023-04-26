"""
Dummy log parser
Doesn't actually parse the log file, just returns defaults.
"""
from argparse import ArgumentParser
from typing import Dict

from cilantro_clients.data_sources.log_parsers.base_log_parser import BaseLogParser


class DummyLogParser(BaseLogParser):
    def __init__(self):
        super(DummyLogParser, self).__init__()

    def get_data(self, log_file) -> Dict:
        """
        Doesn't actually parse the log file, just returns defaults.
        :return:
        """
        return {
            'load': 1,
            'reward': 1,
            'sigma': 1,
        }

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