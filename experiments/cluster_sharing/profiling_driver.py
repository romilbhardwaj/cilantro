"""
    Driver for profiling workloads in a kubernetes cluster.
    -- romilbhardwaj
    -- kirthevasank
"""

# pylint: disable=too-many-statements
# pylint: disable=too-many-locals

import argparse
import asyncio
from datetime import datetime
import logging
import os
# Local
from cilantro.ancillary.info_write_load_utils import write_experiment_info_to_files
from cilantro.backends.alloc_expiration_event_source import AllocExpirationEventSource
from cilantro.backends.grpc.utility_event_source import UtilityEventSource, UTILITY_UPDATE_FIELDS
from cilantro.backends.k8s.kubernetes_manager import KubernetesManager
from cilantro.data_loggers.data_logger_bank import DataLoggerBank
from cilantro.data_loggers.simple_data_logger import SimpleDataLogger
from cilantro.data_loggers.simple_event_logger import SimpleEventLogger
from cilantro.profiling.profiling_policy import ProfilingPolicy
from cilantro.scheduler.cilantroscheduler import CilantroScheduler
from cilantro.workloads.k8s_workload_deployer import K8sWorkloadDeployer
# In Demo
from env_gen import generate_env

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s | %(levelname)-6s | %(name)-40s || %(message)s',
                    datefmt='%m-%d %H:%M:%S'
                    )
logger = logging.getLogger(__name__)

DFLT_ENV_DESCR = 'flat1'
DFLT_REAL_OR_DUMMY = 'dummy'
# DFLT_REAL_OR_DUMMY = 'real'

# For the data logger -----------------------------------------------------
MAX_INMEM_TABLE_SIZE = 1000
REAL_DATA_LOG_WRITE_TO_DISK_EVERY = 60
DUMMY_DATA_LOG_WRITE_TO_DISK_EVERY = 20
REAL_ALLOC_EXPIRATION_TIME = 60 * 2 # Allocate every this many seconds
DUMMY_ALLOC_EXPIRATION_TIME = 10 # Allocate every this many seconds
ALLOC_GRANULARITY = 1 # we cannot assign fractional resources
GRPC_PORT = 10000
LEARNER_DATAPOLL_FREQUENCY = 5 # Learners fetch from data loggers every this many seconds
    # This is used by CilantroScheduler if no learners etc. are specified, if they are not provided
    # expternally.

# For logging and saving results ----------------------------------------------
SCRIPT_TIME_STR = datetime.now().strftime('%m%d%H%M%S')
REPORT_RESULTS_EVERY = 15
MAX_NUM_NODES_FOR_PROFILING = 100

def main():
    """ Main function. """
    # Create args ==================================================================================
    parser = argparse.ArgumentParser(description='Arguments for processing profiled data.')
    parser.add_argument('--env-descr', '-env', type=str, default=DFLT_ENV_DESCR,
                        help='Environment for running experiment.')
    parser.add_argument('--workload-type-to-profile', '-wttp', type=str,
                        help='Workload type to profile.')
    parser.add_argument('--cluster-type', '-clus', type=str,
                        help='Which cluster_type to rund, eks or kind.')
    parser.add_argument('--real-or-dummy', '-rod', type=str, default=DFLT_REAL_OR_DUMMY,
                        help='To run a real or dummy workload.')
    parser.add_argument('--alloc-expiration-time', '-aet', type=int, default=None,
                        help='Allocation expiry time.')
    parser.add_argument('--profiler-resource-allocations', '-pra', type=str, default=None,
                        help='csv values of resource allocations to profile. If not specified,'
                             ', generates a list of allocations to use.')
    args = parser.parse_args()
    workload_type_to_profile = args.workload_type_to_profile
    if args.real_or_dummy == 'real':
        alloc_expiration_time = args.alloc_expiration_time if args.alloc_expiration_time else \
                                REAL_ALLOC_EXPIRATION_TIME
        data_log_write_to_disk_every = REAL_DATA_LOG_WRITE_TO_DISK_EVERY
    else:
        alloc_expiration_time = args.alloc_expiration_time if args.alloc_expiration_time else \
                                DUMMY_ALLOC_EXPIRATION_TIME
        data_log_write_to_disk_every = DUMMY_DATA_LOG_WRITE_TO_DISK_EVERY

    # Create the environment =======================================================================
    env = generate_env(args.env_descr, args.cluster_type, args.real_or_dummy)
    logger.info('Created Env: %s.\n%s', str(env), env.write_to_file(None))

    # Create event sources =========================================================================
    event_queue = asyncio.Queue()
    event_logger = SimpleEventLogger()
    event_loop = asyncio.get_event_loop()
    framework_manager = KubernetesManager(event_queue,
                                          update_loop_sleep_time=1,
                                          dry_run=False)
    event_sources = [UtilityEventSource(output_queue=event_queue, server_port=GRPC_PORT)]
    # Create the allocation expiration event source ------------------------------
    alloc_expiration_event_source = AllocExpirationEventSource(event_queue, alloc_expiration_time)
    event_sources = [alloc_expiration_event_source, *event_sources]

    # Create directories where we will store experimental results =================================
    num_resources = framework_manager.get_cluster_resources()
    profiling_workdir = 'workdirs/prof_%s_%s_%d_%s_%s'%(workload_type_to_profile, args.env_descr,
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
    data_logger_bank = DataLoggerBank(write_to_disk_dir=profiling_workdir,
                                      write_to_disk_every=data_log_write_to_disk_every)
    for leaf_path, leaf in env.leaf_nodes.items():
        data_logger = SimpleDataLogger(
            leaf_path, UTILITY_UPDATE_FIELDS,
            index_fld='event_start_time', max_inmem_table_size=MAX_INMEM_TABLE_SIZE,
            workload_type=leaf.get_workload_info('workload_type'))
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
#     max_num_nodes_for_profiling = min(MAX_NUM_NODES_FOR_PROFILING,
#                                       framework_manager.get_cluster_resources())
    profiler_resource_allocations = args.profiler_resource_allocations
    if profiler_resource_allocations is not None:
        profiler_resource_allocations = [int(p) for p in profiler_resource_allocations.split(",")]
    profiling_policy = ProfilingPolicy(env=env,
                                       resource_quantity=framework_manager.get_cluster_resources(),
                                       leaf_path_to_profile=leaf_path_to_profile,
                                       profiler_resource_allocations=profiler_resource_allocations)
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
    # Initiate learning/reporting etc --------------------------------------------------------------
    data_logger_bank.initiate_write_to_disk_loop()

    # Create the workloads and deploy them =========================================================
    workload_exec = K8sWorkloadDeployer()
    workload_exec.deploy_environment(env)
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

