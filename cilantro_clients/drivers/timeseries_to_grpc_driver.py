'''
A Cilantro client driver which ingests from a timeseries csv and publishes
to grpc. TimeseriesDataSource returns None when there is no data, thus
causing no output intermittently.
'''
import argparse
import logging

from cilantro_clients.cilantro_client.base_cilantro_client import BaseCilantroClient
from cilantro_clients.data_sources.timeseries_data_source import TimeseriesDataSource
from cilantro_clients.publishers.grpc_publisher import GRPCPublisher

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description='A Cilantro client driver which ingests from timeseries and publishes to grpc.')

    # Add parser args
    TimeseriesDataSource.add_args_to_parser(parser)
    GRPCPublisher.add_args_to_parser(parser)
    BaseCilantroClient.add_args_to_parser(parser)

    args = parser.parse_args()
    print(args)

    # Define objects:

    data_source = TimeseriesDataSource(csv_file=args.csv_file,
                                       roll_over=args.roll_over)
    publisher = GRPCPublisher(client_id=args.grpc_client_id,
                              ip=args.grpc_ip,
                              port=args.grpc_port)
    client = BaseCilantroClient(data_source,
                                publisher,
                                poll_frequency=args.poll_frequency)

    client.run_loop()