import logging
from asyncio import Queue
from concurrent import futures

import grpc

from cilantro.backends.grpc.protogen import utility_update_pb2_grpc
from cilantro.backends.grpc.utility_event_source import UtilityMessagingServicer

logging.basicConfig(level=logging.DEBUG)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    q = Queue()
    utility_update_pb2_grpc.add_UtilityMessagingServicer_to_server(
        UtilityMessagingServicer(q, debug_mode=True), server)
    server.add_insecure_port('[::]:10000')
    server.start()
    server.wait_for_termination()

if __name__ == '__main__':
    logging.basicConfig()
    serve()
