# Cilantro driver for local execution and debug
import asyncio
import logging

from cilantro.backends.alloc_expiration_event_source import \
    AllocExpirationEventSource

from cilantro.backends.test.test_backend import DummyEventSource, DummyFrameworkManager

from cilantro.data_loggers.simple_data_logger import SimpleDataLogger
from cilantro.data_loggers.simple_event_logger import SimpleEventLogger
from cilantro.scheduler.cilantroscheduler import CilantroScheduler

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

ALLOC_EXPIR_TIME=5

if __name__ == '__main__':
    dummy_jobs = ['root--user1--cassandra-test']
    policy_str = 'propfair' # 'mmflearn'
    event_queue = asyncio.Queue()
    event_logger = SimpleEventLogger()
    framework_manager = DummyFrameworkManager(event_queue, default_jobs=dummy_jobs, cluster_resources=1)
    # TODO: Configure event sources through config
    # event_sources = [UtilityEventSource(event_queue, server_port=10000),]
    event_sources = [*[DummyEventSource(event_queue, sleep_time=5, app_name=j) for j in dummy_jobs],
                     AllocExpirationEventSource(event_queue, ALLOC_EXPIR_TIME)]
    event_loop = asyncio.get_event_loop()
    cilantro = CilantroScheduler(event_queue=event_queue,
                                 event_logger=event_logger,
                                 policy=policy_str,
                                 framework_manager=framework_manager)
    for s in event_sources:
        event_loop.create_task(s.event_generator())
    try:
        event_loop.run_until_complete(cilantro.scheduler_loop())
    finally:
        event_loop.close()
