"""
Publishes data to stdout.
"""
import logging
import os
from argparse import ArgumentParser
from typing import Dict

import grpc

from cilantro.backends.grpc.protogen import utility_update_pb2_grpc, utility_update_pb2
from cilantro_clients.publishers.base_publisher import BasePublisher, CLIENT_RETCODE_SUCCESS, CLIENT_RETCODE_FAIL

logger = logging.getLogger(__name__)


class GRPCPublisher(BasePublisher):
    def __init__(self,
                 client_id: str,
                 ip: str = None,
                 port: int = None,
                 timeout: float = 1):
        """
        Sends the data over GRPC as a UtilityMessage.
        :param client_id: Used as the publish tag for grpc messages.  If None, reads envvars set by Kubernetes.
        :param ip: IP of GRPC server to publish samples at. If None, reads envvars set by Kubernetes.
        :param port: GRPC server port
        :param timeout: Timeout after which the publish will be failed.
        """
        self.client_id = client_id
        if ip is None:
            # If ip is not specified, assume running inside k8s cluster and find the service
            ip = os.environ['CILANTRO_SERVICE_SERVICE_HOST']
        if port is None:
            port = os.environ['CILANTRO_SERVICE_SERVICE_PORT']
        self.ip = ip
        self.port = port
        self.timeout = timeout
        super(BasePublisher, self).__init__()

    def publish(self, data: Dict) -> [int, str]:
        """
        Publishes data to the output grpc stub and returns a ret code.
        :param data: Dictionary of data to be published
        :return: retcode, 1 if successful, 2 if fail. Also returns an error string
        """
        with grpc.insecure_channel(f'{self.ip}:{self.port}') as channel:
            stub = utility_update_pb2_grpc.UtilityMessagingStub(channel)
            load = float(data['load'])
            alloc = float(data['alloc'])
            reward = float(data['reward'])
            sigma = float(data['sigma'])
            event_start_time = float(data['event_start_time'])
            event_end_time = float(data['event_end_time'])
            debug = str(data['debug'])
            msg = utility_update_pb2.UtilityMessage(app_id=self.client_id,
                                                    load=load,
                                                    alloc=alloc,
                                                    reward=reward,
                                                    sigma=sigma,
                                                    event_start_time=event_start_time,
                                                    event_end_time=event_end_time,
                                                    debug=debug)
            logger.debug(f"Publishing msg: {msg}")
            try:
                stub.PublishUtility(msg,
                                    timeout=self.timeout)
                ret = CLIENT_RETCODE_SUCCESS, None
            except grpc._channel._InactiveRpcError as e:
                if (e._state.code == grpc.StatusCode.DEADLINE_EXCEEDED) or (
                        e._state.code == grpc.StatusCode.UNAVAILABLE):
                    ret = CLIENT_RETCODE_FAIL, e._state.details
                else:
                    raise e
            return ret

    @classmethod
    def add_args_to_parser(self,
                           parser: ArgumentParser):
        parser.add_argument('--grpc-port', '-p', type=int, default=None, help='GRPC Port')
        parser.add_argument('--grpc-ip', '-i', type=str, default=None, help='GRPC IP address')
        parser.add_argument('--grpc-client-id', '-c', type=str, default="TSClient",
                            help='Name to be used for publishing utility messages.')
