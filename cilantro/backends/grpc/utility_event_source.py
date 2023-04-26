import logging
import time
from asyncio import Queue

import grpc

from concurrent import futures

from cilantro.backends.base_event_source import BaseEventSource
from cilantro.backends.grpc.protogen import utility_update_pb2_grpc
from cilantro.backends.grpc.protogen import utility_update_pb2
from cilantro.types.events import UtilityUpdateEvent

logger = logging.getLogger(__name__)

UTILITY_UPDATE_FIELDS = ['load', 'alloc', 'reward', 'sigma', 'event_start_time', 'event_end_time', 'debug']

class UtilityMessagingServicer(utility_update_pb2_grpc.UtilityMessagingServicer):
    # Implements the GRPC Servicer for utility messages from clients
    def __init__(self,
                 event_queue: Queue,
                 debug_mode: bool = False):
        self.event_queues = event_queue
        self.debug_mode = debug_mode
        super(UtilityMessagingServicer, self).__init__()

    def PublishUtility(self,
                       request: utility_update_pb2.UtilityMessage,
                       context):
        event = UtilityUpdateEvent(app_path=request.app_id,
                                   load=request.load,
                                   reward=request.reward,
                                   alloc=request.alloc,
                                   sigma=request.sigma,
                                   event_start_time=request.event_start_time,
                                   event_end_time=request.event_end_time,
                                   timestamp=time.time(),
                                   debug=request.debug)
        self.event_queues.put_nowait(event)
        if self.debug_mode:
            logger.debug(f"Got event: {str(event)}")
        return utility_update_pb2.UtilityAck(retcode=0)

class UtilityEventSource(BaseEventSource):
    '''Runs a utility grpc server to get data'''
    def __init__(self,
                 output_queue: Queue,
                 server_port: int):
        super(UtilityEventSource, self).__init__(output_queue)
        self.server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=10))
        utility_update_pb2_grpc.add_UtilityMessagingServicer_to_server(
            UtilityMessagingServicer(self.output_queue), self.server)
        self.server.add_insecure_port(f'[::]:{server_port}')

    async def event_generator(self):
        '''
        Long running loop that generates events indefinitely
        :return:
        '''
        await self.server.start()
        await self.server.wait_for_termination()

    def __del__(self):
        self.server.stop(grace=0)