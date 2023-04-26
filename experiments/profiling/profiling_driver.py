"""
    Driver for profiling workloads in a kubernetes cluster.
    -- romilbhardwaj
"""

# pylint: disable=import-error
# pylint: disable=too-many-statements

import argparse
import asyncio
from datetime import datetime
import logging
import os
# Local
from cilantro.ancillary.info_write_load_utils import write_experiment_info_to_files
from cilantro.backends.alloc_expiration_event_source import AllocExpirationEventSource
from cilantro.backends.grpc.utility_event_source import UtilityEventSource
from cilantro.backends.k8s.kubernetes_manager import KubernetesManager
from cilantro.backends.test.test_backend import DummyFrameworkManager, DummyEventSource
from cilantro.data_loggers.data_logger_bank import DataLoggerBank
from cilantro.data_loggers.simple_data_logger import SimpleDataLogger
from cilantro.data_loggers.simple_event_logger import SimpleEventLogger
from cilantro.profiling.profiling_policy import ProfilingPolicy
from cilantro.scheduler.cilantroscheduler import CilantroScheduler
from cilantro.workloads.k8s_workload_deployer import K8sWorkloadDeployer
# In Demo
from env_demo import generate_env

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)-6s | %(name)-40s || %(message)s',
                    datefmt='%m-%d %H:%M:%S'
                    )
logger = logging.getLogger(__name__)


ENV_DESCR = 'simple'

# For the data logger -----------------------------------------------------
MAX_INMEM_TABLE_SIZE = -1
MAX_INMEM_TABLE_SIZE = 1000
DATA_LOG_WRITE_TO_DISK_EVERY = 30
ALLOC_GRANULARITY = 1 # we cannot assign fractional resources

# Other parameters
ASYNC_SLEEP_TIME = 0.5
SLEEP_TIME_BETWEEN_DATA_REPOLLS = 1.1
ALLOC_EXPIR_TIME = 6 # Allocate every this many seconds
GRPC_PORT = 10000
LEARNER_DATAPOLL_FREQUENCY = 5 # Learners fetch from data loggers every this many seconds
DEBUG_MODE = False
DUMMY_NUM_RESOURCES = 12

# For logging and saving results ----------------------------------------------
SCRIPT_TIME_STR = datetime.now().strftime('%m%d%H%M%S')
REPORT_RESULTS_EVERY = 15


def main():
    """ Main function. """
    # Create args ==================================================================================
    parser = argparse.ArgumentParser(description='Arguments for processing profiled data.')
    parser.add_argument('--workload-type-to-profile', '-wttp', type=str,
                        help='Workload type to profile.')
    parser.add_argument('--cluster-type', '-clus', type=str,
                        help='Which cluster_type to rund, eks or kind.')
    args = parser.parse_args()
    workload_type_to_profile = args.workload_type_to_profile

    # Create the environment =======================================================================
    env = generate_env(args.cluster_type)
    env_jobs = env.get_leaf_node_paths()
    logging.info('Created Env: %s.\n%s', str(env), env.write_to_file(None))

    # Create event sources =========================================================================
    event_queue = asyncio.Queue()
    event_logger = SimpleEventLogger()
    event_loop = asyncio.get_event_loop()
    if not DEBUG_MODE:
        framework_manager = KubernetesManager(event_queue,
                                              update_loop_sleep_time=1,
                                              dry_run=False)
        event_sources = [UtilityEventSource(output_queue=event_queue, server_port=GRPC_PORT)]
    else:
        app_event_sources_dict = {
            j: DummyEventSource(event_queue, sleep_time=ASYNC_SLEEP_TIME, app_name=j)
            for j in env_jobs}
        framework_manager = DummyFrameworkManager(event_queue, default_jobs=env_jobs,
                                                  cluster_resources=DUMMY_NUM_RESOURCES,
                                                  alloc_granularity=ALLOC_GRANULARITY)
        event_sources = [aes for _, aes in app_event_sources_dict.items()]
    # Create the allocation expiration event source -----------------------------------------------
    alloc_expiration_event_source = AllocExpirationEventSource(event_queue, ALLOC_EXPIR_TIME)
    event_sources = [alloc_expiration_event_source, *event_sources]

    # Create directories where we will store experimental results =================================
    num_resources = framework_manager.get_cluster_resources()
    profiling_workdir = 'profiledirs/%s_%s_%d_%s_%s'%(workload_type_to_profile, ENV_DESCR,
                                                      num_resources,
                                                      args.cluster_type, SCRIPT_TIME_STR)
    if not os.path.exists(profiling_workdir):
        os.makedirs(profiling_workdir, exist_ok=True)
    # Write experimental information to file before commencing experiments ------------------------
    experiment_info = {}
    experiment_info['resource_quantity'] = num_resources
    experiment_info['alloc_granularity'] = framework_manager.get_alloc_granularity()
    write_experiment_info_to_files(profiling_workdir, env, experiment_info)

    # Create data loggers for each node ===========================================================
    data_logger_bank = DataLoggerBank()
    for leaf_path, leaf in env.leaf_nodes.items():
        data_logger = SimpleDataLogger(
            leaf_path, ['load', 'alloc', 'reward', 'sigma', 'event_start_time', 'event_end_time'],
            index_fld='event_start_time', max_inmem_table_size=MAX_INMEM_TABLE_SIZE,
            workload_type=leaf.get_workload_info('workload_type'),
            disk_dir=profiling_workdir, write_to_disk_every=DATA_LOG_WRITE_TO_DISK_EVERY)
        data_logger_bank.register(leaf_path, data_logger)


    # Decide which leaf to profile ===============================================================
    leaf_path_to_profile = None
    for leaf_path, leaf in env.leaf_nodes.items():
        if leaf.get_workload_info('workload_type') == workload_type_to_profile:
            leaf_path_to_profile = leaf_path
            break
    logger.info('Profiling workload %s with leaf: %s.', workload_type_to_profile,
                leaf_path_to_profile)

    # Create profiling policy ====================================================================
    profiling_policy = ProfilingPolicy(env=env,
                                       resource_quantity=framework_manager.get_cluster_resources(),
                                       leaf_path_to_profile=leaf_path_to_profile)
    profiling_policy.initialise()
    logger.info('Policy initialised')

    # Pass learner bank and time series model to the scheduler =====================================
    cilantro = CilantroScheduler(event_queue=event_queue,
                                 framework_manager=framework_manager,
                                 event_logger=event_logger,
                                 env=env,
                                 policy=profiling_policy,
                                 data_logger_bank=data_logger_bank,
                                 learner_bank=None,
                                 performance_recorder_bank=None,
                                 load_forecaster_bank=None,
                                 learner_datapoll_frequency=LEARNER_DATAPOLL_FREQUENCY)


    # Create the workloads and deploy them =========================================================
    if not DEBUG_MODE:
        workload_exec = K8sWorkloadDeployer()
        workload_exec.deploy_environment(env)
    else:
        logger.info("Debug mode - workload deployment is no-op")
    logger.info('Workload deployed!')
    # Create event sources =========================================================================
    for s in event_sources:
        event_loop.create_task(s.event_generator())
    try:
        event_loop.run_until_complete(cilantro.scheduler_loop())
    finally:
        event_loop.close()


if __name__ == '__main__':
    main()

