"""
Driver for cilantro that runs inside a kubernetes scheduler.
Populates the environment by reading kubernetes state.
"""

import asyncio
import logging
import time
import numpy as np
# Local
from cilantro.backends.base_event_source import BaseEventSource
from cilantro.backends.alloc_expiration_event_source import AllocExpirationEventSource
from cilantro.backends.base_framework_manager import BaseFrameworkManager
from cilantro.backends.grpc.utility_event_source import UtilityEventSource
from cilantro.backends.k8s.kubernetes_manager import KubernetesManager
from cilantro.backends.test.test_backend import DummyFrameworkManager
from cilantro.core.henv import LinearLeafNode, InternalNode, TreeEnvironment
from cilantro.core.performance_recorder import PerformanceRecorder, PerformanceRecorderBank
from cilantro.data_loggers.data_logger_bank import DataLoggerBank
from cilantro.data_loggers.simple_data_logger import SimpleDataLogger
from cilantro.data_loggers.simple_event_logger import SimpleEventLogger
from cilantro.learners.base_learner import BaseLearner
from cilantro.learners.ibtree import IntervalBinaryTree
from cilantro.learners.learner_bank import LearnerBank
from cilantro.policies.prop_fairness import PropFairness
from cilantro.policies.mmflearn import MMFLearn
from cilantro.scheduler.cilantroscheduler import CilantroScheduler
from cilantro.timeseries.ts_forecaster_bank import TSForecasterBank
from cilantro.timeseries.ts_base_learner import TSBaseLearner
from cilantro.types.events import AppAddEvent, EventTypes, UtilityUpdateEvent

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s | %(levelname)-6s | %(name)-40s || %(message)s',
                    datefmt='%m-%d %H:%M:%S'
                    )
logger = logging.getLogger(__name__)

POLICY_NAME = 'propfair'
# POLICY_NAME = 'mmflearn'
# POLICY_NAME = 'mmf'

INT_UPPER_BOUND = 0.03
LIP_CONST = 2

GRPC_PORT = 10000

ALLOC_GRANULARITY = 1 # we cannot assign fractional resources


# Other parameters
ASYNC_SLEEP_TIME = 0.5
SLEEP_TIME_BETWEEN_DATA_REPOLLS = 1.1
ALLOC_EXPIR_TIME = 10 # Allocate every this many seconds


def generate_env():
    """ Generates a treenvironment with a root."""
    root = InternalNode('root')
    env = TreeEnvironment(root, 1)
    return env


def main():
    """ Main function. """
    # Create the environment =======================================================================
    env = generate_env()
    print('Created Env: %s'%(env))

    # Create event loggers and framework managers ==================================================
    event_queue = asyncio.Queue()
    event_logger = SimpleEventLogger()
    event_loop = asyncio.get_event_loop()
    framework_manager = KubernetesManager(event_queue,
                                          update_loop_sleep_time=1,
                                          dry_run=False)
    # Create event sources
    util_event_source = UtilityEventSource(output_queue=event_queue, server_port=GRPC_PORT)
    alloc_expiration_event_source = AllocExpirationEventSource(event_queue, ALLOC_EXPIR_TIME)
    event_sources = [alloc_expiration_event_source,
                     util_event_source]

    # Create banks
    load_forecaster_bank = TSForecasterBank()
    data_logger_bank = DataLoggerBank()
    learner_bank = LearnerBank()
    performance_recorder_bank = PerformanceRecorderBank(
        resource_quantity=framework_manager.get_cluster_resources(),
        alloc_granularity=framework_manager.get_alloc_granularity())

    # Create policy ================================================================================
    if POLICY_NAME == 'propfair':
        policy = PropFairness(env=env, resource_quantity=framework_manager.get_cluster_resources(),
                              load_forecaster_bank=load_forecaster_bank, alloc_granularity=1)
    elif POLICY_NAME == 'mmflearn':
        policy = MMFLearn(env=env, resource_quantity=framework_manager.get_cluster_resources(),
                          load_forecaster_bank=load_forecaster_bank, learner_bank=learner_bank,
                          alloc_granularity=framework_manager.get_alloc_granularity())
    else:
        raise ValueError('Unknown policy_name %s.'%(POLICY_NAME))
    policy.initialise()


    # Pass learner bank and time series model to the scheduler =====================================
    cilantro = CilantroScheduler(event_queue=event_queue,
                                 framework_manager=framework_manager,
                                 event_logger=event_logger,
                                 env=env,
                                 policy=policy,
                                 data_logger_bank=data_logger_bank,
                                 learner_bank=learner_bank,
                                 load_forecaster_bank=load_forecaster_bank,
                                 performance_recorder_bank=performance_recorder_bank)


    # Create event sources =========================================================================
    for s in event_sources:
        event_loop.create_task(s.event_generator())
    try:
        event_loop.run_until_complete(cilantro.scheduler_loop())
    finally:
        event_loop.close()


if __name__ == '__main__':
    main()
