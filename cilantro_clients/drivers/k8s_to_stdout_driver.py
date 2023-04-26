import argparse
import logging
import random

from cilantro_clients.cilantro_client.base_cilantro_client import BaseCilantroClient
from cilantro_clients.data_sources.dummy_data_source import DummyDataSource
from cilantro_clients.data_sources.k8s_proportional_data_source import K8sProportionalDataSource
from cilantro_clients.publishers.stdout_publisher import StdoutPublisher

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='A Cilantro client driver which ingests from k8s and publishes to stdout.')

    # Add parser args
    K8sProportionalDataSource.add_args_to_parser(parser)
    StdoutPublisher.add_args_to_parser(parser)
    BaseCilantroClient.add_args_to_parser(parser)

    args = parser.parse_args()

    # Define metrics as a function of allocation ======================================================
    # Change these functions for testing.
    load_fn = lambda alloc: random.randint(0, 10)   # This can be read from a timeseries file
    alloc_fn = lambda alloc: alloc  # Change if reported allocation is different from actual
    reward_fn = lambda alloc: alloc # Reward is proportional to allocation
    sigma_fn = lambda alloc: 1

    # Define objects:

    data_source = K8sProportionalDataSource(args.k8s_app_name,
                                            load_fn=load_fn,
                                            alloc_fn=alloc_fn,
                                            reward_fn=reward_fn,
                                            sigma_fn=sigma_fn,
                                            namespace=args.k8s_namespace)
    publisher = StdoutPublisher()
    client = BaseCilantroClient(data_source,
                                publisher,
                                poll_frequency=args.poll_frequency)
    client.run_loop()