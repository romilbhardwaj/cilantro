'''
    A Cilantro client driver which reads log folders written by YCSB benchmark and publishes
    to grpc. TimeseriesDataSource returns None when there is no data, thus
    causing no output intermittently.
'''

import argparse
import logging

from cilantro_clients.cilantro_client.base_cilantro_client import BaseCilantroClient
from cilantro_clients.data_sources.log_parsers.wrk2_log_parser import \
    WrkLogParser
from cilantro_clients.data_sources.log_parsers.ycsb_log_parser import YCSBLogParser
from cilantro_clients.data_sources.logfolder_data_source import LogFolderDataSource
from cilantro_clients.publishers.grpc_publisher import GRPCPublisher
from cilantro_clients.publishers.stdout_publisher import StdoutPublisher


def main():
    """ Main function. """
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        description='A Cilantro client driver which ingests from ycsb log folders and ' + \
                    'publishes to grpc.'
        )
    # Add parser args
    WrkLogParser.add_args_to_parser(parser)
    LogFolderDataSource.add_args_to_parser(parser)
    StdoutPublisher.add_args_to_parser(parser)
    GRPCPublisher.add_args_to_parser(parser)
    BaseCilantroClient.add_args_to_parser(parser)
    args = parser.parse_args()
    print(args)

    # ====== Specify your logparser here ======
    # log_parser = DummyLogParser()
    # log_parser = YCSBLogParser(slo_latency=args.slo_latency)
    log_parser = WrkLogParser()

    # Define objects:
    data_source = LogFolderDataSource(log_dir_path=args.log_folder_path,
                                      log_parser=log_parser,
                                      log_extension=args.log_extension)
    # publisher = StdoutPublisher()
    publisher = [GRPCPublisher(client_id=args.grpc_client_id,
                              ip=args.grpc_ip,
                              port=args.grpc_port),
                 StdoutPublisher()]

    client = BaseCilantroClient(data_source,
                                publisher,
                                poll_frequency=args.poll_frequency)
    client.run_loop()

if __name__ == '__main__':
    main()

