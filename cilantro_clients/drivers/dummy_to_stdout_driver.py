'''
A Cilantro client driver which ingests from dummy_data_source and publishes
to stdout. This is the most basic form of a cilantro driver.
'''
import argparse
import logging

from cilantro_clients.cilantro_client.base_cilantro_client import BaseCilantroClient
from cilantro_clients.data_sources.dummy_data_source import DummyDataSource
from cilantro_clients.publishers.stdout_publisher import StdoutPublisher

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='A Cilantro client driver which ingests from dummy_data_source and publishes to stdout. This is the most basic form of a cilantro driver.')

    # Add parser args
    DummyDataSource.add_args_to_parser(parser)
    StdoutPublisher.add_args_to_parser(parser)
    BaseCilantroClient.add_args_to_parser(parser)

    args = parser.parse_args()

    # Define objects:

    data_source = DummyDataSource()
    publisher = StdoutPublisher()
    client = BaseCilantroClient(data_source,
                                publisher,
                                poll_frequency=args.poll_frequency)
    client.run_loop()