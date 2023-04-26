"""
This cilantro client looks at the current resource allocation in kubernetes to a specified workload, and returns
the metrics as a function of the allocation. This is useful for debugging and testing.
"""
import argparse
import logging
import random

from cilantro_clients.cilantro_client.base_cilantro_client import BaseCilantroClient
from cilantro_clients.data_sources.k8s_proportional_data_source import K8sProportionalDataSource
from cilantro_clients.publishers.grpc_publisher import GRPCPublisher

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='A Cilantro client driver which ingests allocations from k8s and publishes metrics to stdout.')

    # Add parser args
    K8sProportionalDataSource.add_args_to_parser(parser)
    GRPCPublisher.add_args_to_parser(parser)
    BaseCilantroClient.add_args_to_parser(parser)

    args = parser.parse_args()

    # Define metrics as a function of allocation
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
    publisher = GRPCPublisher(client_id=args.grpc_client_id,
                              ip=args.grpc_ip,
                              port=args.grpc_port)
    client = BaseCilantroClient(data_source,
                                publisher,
                                poll_frequency=args.poll_frequency)
    client.run_loop()
