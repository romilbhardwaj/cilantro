"""
This cilantro client fetches metrics from YCSB logs and publishes them to the scheduler.
Uses a k8s allocation source to append alloc data to the reported metrics.
"""
import argparse
import logging
import random

from cilantro_clients.alloc_sources.k8s_alloc_source import K8sAllocSource
from cilantro_clients.cilantro_client.base_cilantro_client import BaseCilantroClient
from cilantro_clients.data_sources.k8s_proportional_data_source import K8sProportionalDataSource
from cilantro_clients.data_sources.log_parsers.ycsb_log_parser import YCSBLogParser
from cilantro_clients.data_sources.logfolder_data_source import LogFolderDataSource
from cilantro_clients.publishers.grpc_publisher import GRPCPublisher
from cilantro_clients.publishers.stdout_publisher import StdoutPublisher

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s | %(levelname)-6s | %(name)-40s || %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    logger = logging.getLogger(__name__)

    parser = argparse.ArgumentParser(
        description='A Cilantro client driver which fetches metrics from YCSB logs and publishes them to the scheduler.')

    # Add parser args
    LogFolderDataSource.add_args_to_parser(parser)
    K8sAllocSource.add_args_to_parser(parser)
    GRPCPublisher.add_args_to_parser(parser)
    BaseCilantroClient.add_args_to_parser(parser)
    parser.add_argument('--slo-latency', '-lat', type=float,
                        help='The latency for the SLO.')

    args = parser.parse_args()
    logger.info(f"Command line args: {args}")

    # ====== Specify your logparser here ======
    # log_parser = DummyLogParser()
    log_parser = YCSBLogParser(slo_latency=args.slo_latency)

    alloc_source = K8sAllocSource(app_name=args.app_name)   # This is no longer needed if allocation is written to file by the client.

    # Define objects:
    data_source = LogFolderDataSource(log_dir_path=args.log_folder_path,
                                      log_parser=log_parser,
                                      log_extension=args.log_extension)
    grpcpublisher = GRPCPublisher(client_id=args.grpc_client_id,
                              ip=args.grpc_ip,
                              port=args.grpc_port)
    stdoutpublisher = StdoutPublisher()
    client = BaseCilantroClient(data_source,
                                publisher=[grpcpublisher, stdoutpublisher],
                                poll_frequency=args.poll_frequency,
                                alloc_source=alloc_source)
    client.run_loop()
