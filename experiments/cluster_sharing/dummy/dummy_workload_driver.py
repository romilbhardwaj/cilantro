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
# Local
from cilantro_clients.cilantro_client.base_cilantro_client import BaseCilantroClient
from cilantro_clients.publishers.grpc_publisher import GRPCPublisher
# demo
from dummy_data_source import AllocationBasedDataSource, LinearAllocationBasedDataSource, \
                              TanhAllocationBasedDataSource, VPIAllocationBasedDataSource, \
                              LogisticAllocationBasedDataSource

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)-6s | %(name)-40s || %(message)s',
                    datefmt='%m-%d %H:%M:%S'
                    )
logger = logging.getLogger(__name__)


def main():
    """ Main function. """
    # Add parser args -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(
        description=('A Cilantro client driver which ingests allocations from k8s and ' +
                     'publishes metrics to stdout.'))
    AllocationBasedDataSource.add_args_to_parser(parser)
    GRPCPublisher.add_args_to_parser(parser)
    BaseCilantroClient.add_args_to_parser(parser)
    args = parser.parse_args()

    # Create the dummy data source -----------------------------------------------------------
    if args.workload_type == 'dummydataserving':
        coeff = 750.42
        data_source = TanhAllocationBasedDataSource(args.k8s_app_name,
                                                    namespace=args.k8s_namespace,
                                                    coeff=coeff,
                                                    load_file=args.load_file)
    elif args.workload_type == 'dummydatacaching':
        coeff = 623.45
        poly_factor = 1
        data_source = VPIAllocationBasedDataSource(args.k8s_app_name,
                                                   namespace=args.k8s_namespace,
                                                   coeff=coeff,
                                                   load_file=args.load_file,
                                                   poly_factor=poly_factor)
    elif args.workload_type == 'dummyinmemoryanalytics':
        coeff = 919.67
        poly_factor = 1
        data_source = VPIAllocationBasedDataSource(args.k8s_app_name,
                                                   namespace=args.k8s_namespace,
                                                   coeff=coeff,
                                                   load_file=args.load_file,
                                                   poly_factor=poly_factor)
    elif args.workload_type == 'dummywebserving':
        coeff = 400
        sigma_max = 0.5
        data_source = LinearAllocationBasedDataSource(args.k8s_app_name,
                                                      namespace=args.k8s_namespace,
                                                      coeff=coeff,
                                                      sigma_max=sigma_max,
                                                      load_file=args.load_file)
    elif args.workload_type == 'dummydataanalytics':
        coeff = 1129.92
        bias = 9.1
        data_source = LogisticAllocationBasedDataSource(args.k8s_app_name,
                                                        namespace=args.k8s_namespace,
                                                        coeff=coeff,
                                                        bias=bias,
                                                        load_file=args.load_file)
    elif args.workload_type == 'dummywebsearch':
        coeff = 1761.53
        bias = 4.9
        data_source = LogisticAllocationBasedDataSource(args.k8s_app_name,
                                                        namespace=args.k8s_namespace,
                                                        coeff=coeff,
                                                        bias=bias,
                                                        load_file=args.load_file)
    else:
        raise ValueError('Unknown value for args.workload_type=%s.'%(args.workload_type))

    # Create publisher and client and then run loop ------------------------------------------
    publisher = GRPCPublisher(client_id=args.grpc_client_id,
                              ip=args.grpc_ip,
                              port=args.grpc_port)
    client = BaseCilantroClient(data_source,
                                publisher,
                                poll_frequency=args.poll_frequency)
    client.run_loop()


if __name__ == '__main__':
    main()

