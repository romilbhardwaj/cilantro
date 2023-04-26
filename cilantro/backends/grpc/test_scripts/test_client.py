import argparse
import grpc
import logging

from cilantro.backends.grpc.protogen import utility_update_pb2_grpc, utility_update_pb2

DEFAULT_IP = 'localhost'
DEFAULT_PORT = 10000

def parseargs():
    parser = argparse.ArgumentParser(description='GRPC dummy client that generates and sends utility messages.')
    parser.add_argument('--port', '-p', type=int, default=DEFAULT_PORT, help='GRPC Port')
    parser.add_argument('--ip', '-i', type=str, default=DEFAULT_IP, help='GRPC IP addresss')
    args = parser.parse_args()
    return args

def run_client(ip, port):
    with grpc.insecure_channel(f'{ip}:{port}') as channel:
        stub = utility_update_pb2_grpc.UtilityMessagingStub(channel)
        print("-------------- PublishUtility --------------")
        ret = stub.PublishUtility(utility_update_pb2.UtilityMessage(app_id="app1",
                                                                    load=10,
                                                                    alloc=1,
                                                                    reward=0,
                                                                    sigma=1,
                                                                    event_start_time=0,
                                                                    event_end_time=1,
                                                                    debug="Hello."
                                                                    ))
        print(f"Got retcode {ret.retcode}")


if __name__ == '__main__':
    logging.basicConfig()
    args = parseargs()
    logging.info(args)
    run_client(args.ip, args.port)