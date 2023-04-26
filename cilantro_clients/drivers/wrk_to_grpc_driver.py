"""
    This cilantro client fetches metrics wrk logs and publishes them to the scheduler.
"""

import argparse
import logging
# cilantro
from cilantro_clients.cilantro_client.base_cilantro_client import BaseCilantroClient
from cilantro_clients.data_sources.log_parsers.cray_log_parser import CrayLogParser
from cilantro_clients.data_sources.log_parsers.wrk2_log_parser import \
    WrkLogParser
from cilantro_clients.data_sources.logfolder_data_source import LogFolderDataSource
from cilantro_clients.publishers.grpc_publisher import GRPCPublisher
from cilantro_clients.publishers.stdout_publisher import StdoutPublisher

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)-6s | %(name)-40s || %(message)s',
                    datefmt='%m-%d %H:%M:%S')
logger = logging.getLogger(__name__)


def main():
    """ main function. """
    parser = argparse.ArgumentParser(
        description='A Cilantro client which fetches wrk logs and publishes them.')
    # Add parser args
    LogFolderDataSource.add_args_to_parser(parser)
    GRPCPublisher.add_args_to_parser(parser)
    BaseCilantroClient.add_args_to_parser(parser)
    WrkLogParser.add_args_to_parser(parser)
    args = parser.parse_args()
    logger.info(f"Command line args: {args}")

    # log_parser = DummyLogParser()
    log_parser = WrkLogParser()

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
                                poll_frequency=args.poll_frequency)
    client.run_loop()


if __name__ == '__main__':
    main()
