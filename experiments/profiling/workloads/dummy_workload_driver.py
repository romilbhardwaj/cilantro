"""
    This cilantro client looks at the current resource allocation in kubernetes to a specified
    workload, and returns the metrics as a function of the allocation. This is useful for debugging
    and testing.
    -- romilbhardwaj
    -- kirthevasank
"""

# pylint: disable=import-error

import argparse
import logging

from cilantro_clients.cilantro_client.base_cilantro_client import BaseCilantroClient
from cilantro_clients.publishers.grpc_publisher import GRPCPublisher
# Local
from k8s_proportional_data_source import K8sProportionalDataSource

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)-6s | %(name)-40s || %(message)s',
                    datefmt='%m-%d %H:%M:%S'
                    )
logger = logging.getLogger(__name__)


def main():
    """ Main function. """
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description=('A Cilantro client driver which ingests allocations from k8s and ' +
                     'publishes metrics to stdout.'))

    # Add parser args
    K8sProportionalDataSource.add_args_to_parser(parser)
    GRPCPublisher.add_args_to_parser(parser)
    BaseCilantroClient.add_args_to_parser(parser)

    args = parser.parse_args()

    # Define objects:
    data_source = K8sProportionalDataSource(args.k8s_app_name,
                                            namespace=args.k8s_namespace,
                                            load_file=args.load_file)
    publisher = GRPCPublisher(client_id=args.grpc_client_id,
                              ip=args.grpc_ip,
                              port=args.grpc_port)
    client = BaseCilantroClient(data_source,
                                publisher,
                                poll_frequency=args.poll_frequency)
    client.run_loop()


if __name__ == '__main__':
    main()

