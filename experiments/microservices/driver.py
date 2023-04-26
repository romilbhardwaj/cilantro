"""
    Loads a cilantro scheduler by first loading profiled data and then running the policy.
    -- kirthevasank
    -- romilbhardwaj
"""

# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=too-many-branches

import argparse
import asyncio
import os
import time
from datetime import datetime
import logging
# Local
from cilantro.ancillary.info_write_load_utils import write_experiment_info_to_files
from cilantro.backends.alloc_expiration_event_source import AllocExpirationEventSource
from cilantro.backends.grpc.utility_event_source import UtilityEventSource
from cilantro.backends.k8s.kubernetes_manager import KubernetesManager
from cilantro.core.ms_performance_recorder import MSPerformanceRecorder, MSPerformanceRecorderBank
from cilantro.data_loggers.data_logger_bank import DataLoggerBank
from cilantro.data_loggers.simple_data_logger import SimpleDataLogger
from cilantro.data_loggers.simple_event_logger import SimpleEventLogger
from cilantro.learners.p99_learner import P99Learner
from cilantro.learners.gp import GP
from cilantro.learners.learner_bank import LearnerBank
from cilantro.policies.prop_fairness import PropFairness
from cilantro.policies.ms_evo_opt import MSEvoOpt
from cilantro.policies.ms_interleaved_exploration import MSInterleavedExploration
from cilantro.policies.ucb_opt import UCBOpt
from cilantro.scheduler.cilantroscheduler import CilantroScheduler
from cilantro.timeseries.ts_forecaster_bank import TSForecasterBank
from cilantro.timeseries.ts_base_learner import TSBaseLearner
# In Demo
from env_gen import generate_env, HOTELRES_MICROSERVICES

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s | %(levelname)-6s | %(name)-40s || %(message)s',
                    datefmt='%m-%d %H:%M:%S'
                    )
logger = logging.getLogger(__name__)

# DFLT_REAL_OR_DUMMY = 'dummy'
DFLT_REAL_OR_DUMMY = 'real'

# For the data logger -----------------------------------------------------
MAX_INMEM_TABLE_SIZE = 1000
DUMMY_DATA_LOG_WRITE_TO_DISK_EVERY = 20
REAL_DATA_LOG_WRITE_TO_DISK_EVERY = 60
REAL_ALLOC_EXPIRATION_TIME = 60 * 2 # Allocate every this many seconds
DUMMY_ALLOC_EXPIRATION_TIME = 10 # Allocate every this many seconds
ALLOC_GRANULARITY = 1 # we cannot assign fractional resources
GRPC_PORT = 10000
LEARNER_DATAPOLL_FREQUENCY = 5 # Learners fetch from data loggers every this many seconds
    # This is used by CilantroScheduler if no learners etc. are specified, if they are not provided
    # expternally.
NUM_PARALLEL_TRAINING_THREADS = 7
SLEEP_TIME_BETWEEN_TRAINING = 5
APP_CLIENT_KEY = 'hr-client'    # This is set in cilantro-hr-client.yaml

# For logging and saving results ----------------------------------------
SCRIPT_TIME_STR = datetime.now().strftime('%m%d%H%M%S')
REPORT_RESULTS_EVERY = 60 * 4
# REPORT_RESULTS_EVERY = 17
DFLT_NUM_EVO_OPT_ITERS = 4000


def main():
    """ Main function. """
    # Parse args and create directories ============================================================
    parser = argparse.ArgumentParser(description='Arguments for running experiment.')
    parser.add_argument('--env-descr', '-env', type=str,
                        help='Environment for running experiment.', default='hotelres')
    parser.add_argument('--cluster-type', '-clus', type=str,
                        help='Which cluster_type to run, eks or kind.')
    parser.add_argument('--real-or-dummy', '-rod', type=str, default=DFLT_REAL_OR_DUMMY,
                        help='To run a real or dummy workload.')
#     parser.add_argument('--profiled-info-dir', '-pid', type=str,
#                         help='Directory which has the profiled data saved.')
    parser.add_argument('--policy', '-pol', type=str,
                        help='Which policy to run.')
    parser.add_argument('--num-iters-for-evo-opt', '-evoiters', type=int,
                        default=DFLT_NUM_EVO_OPT_ITERS, help='Num iters for evolutionary opt.')
    args = parser.parse_args()
    if args.real_or_dummy == 'real':
        alloc_expiration_time = REAL_ALLOC_EXPIRATION_TIME
        data_log_write_to_disk_every = REAL_DATA_LOG_WRITE_TO_DISK_EVERY
    else:
        alloc_expiration_time = DUMMY_ALLOC_EXPIRATION_TIME
        data_log_write_to_disk_every = DUMMY_DATA_LOG_WRITE_TO_DISK_EVERY

    # Create the environment and other initial set up ==============================================
    env = generate_env(args.env_descr, args.cluster_type, args.real_or_dummy)
    alloc_leaf_order = sorted(list(env.leaf_nodes))
    logger.info('Created Env: %s.\n%s', str(env), env.write_to_file(None))
    logging.info('Created Env: %s,\n%s\n', str(env), env.write_to_file(None))
#     # Create directories for logging results
#     profiled_info_bank = ProfiledInfoBank(args.profiled_info_dir)
#     logging.info('Loading profiled information from %s.', args.profiled_info_dir)
#     logging.info(profiled_info_bank.display_profiled_information_for_env(env))

    # Create event loggers, framework managers, and event sources ==================================
    event_queue = asyncio.Queue()
    event_logger = SimpleEventLogger()
    event_loop = asyncio.get_event_loop()
    framework_manager = KubernetesManager(event_queue,
                                          update_loop_sleep_time=1,
                                          dry_run=False)
    event_sources = [UtilityEventSource(output_queue=event_queue, server_port=GRPC_PORT)]
    # Create the allocation expiration event source -----------------------------------------------
    alloc_expiration_event_source = AllocExpirationEventSource(event_queue, alloc_expiration_time)
    event_sources = [alloc_expiration_event_source, *event_sources]

    # Create directories where we will store experimental results =================================
    num_resources = framework_manager.get_cluster_resources()
    experiment_workdir = 'workdirs/%s_%s_%d_%s_%s'%(args.policy, args.env_descr, num_resources,
                                                    args.cluster_type, SCRIPT_TIME_STR)
    if not os.path.exists(experiment_workdir):
        os.makedirs(experiment_workdir, exist_ok=True)
    save_results_file_path = os.path.join(experiment_workdir, 'in_run_results.p')
    # Write experimental information to file before commencing experiments ------------------------
    experiment_info = {}
    experiment_info['resource_quantity'] = num_resources
    experiment_info['alloc_granularity'] = framework_manager.get_alloc_granularity()
    write_experiment_info_to_files(experiment_workdir, env, experiment_info)

    # Create data loggers, learners, time_series forcaseters and performance recorder for each leaf
    # node ========================================================================================
    data_logger_bank = DataLoggerBank(write_to_disk_dir=experiment_workdir,
                                      write_to_disk_every=data_log_write_to_disk_every)
    learner_bank = LearnerBank(num_parallel_training_threads=NUM_PARALLEL_TRAINING_THREADS,
                               sleep_time_between_trains=SLEEP_TIME_BETWEEN_TRAINING)
    load_forecaster_bank = TSForecasterBank(
        num_parallel_training_threads=NUM_PARALLEL_TRAINING_THREADS,
        sleep_time_between_trains=SLEEP_TIME_BETWEEN_TRAINING)
    reward_forecaster_bank = TSForecasterBank(
        num_parallel_training_threads=NUM_PARALLEL_TRAINING_THREADS,
        sleep_time_between_trains=SLEEP_TIME_BETWEEN_TRAINING)
    ms_performance_recorder_bank = MSPerformanceRecorderBank(
        resource_quantity=framework_manager.get_cluster_resources(),
        alloc_granularity=framework_manager.get_alloc_granularity(),
        report_results_every=REPORT_RESULTS_EVERY, save_file_name=save_results_file_path,
        report_results_descr=args.policy,
        )
    # Create dataloggers, reward_forecasters, and performance recorders for the app_client.
    # =====================================================================================
    # the data logger --------------------------------------------------------------------------
    app_client_data_logger = SimpleDataLogger(
        APP_CLIENT_KEY, ['load', 'DEBUG.avg_latency', 'DEBUG.p99', 'DEBUG.stddev_latency',
                         'DEBUG.allocs', 'event_start_time', 'event_end_time'],
        index_fld='event_start_time', max_inmem_table_size=MAX_INMEM_TABLE_SIZE)
        # DEBUG.allocs is a dictionary of {microservice: num_cpus} e.g. {root--consul: 8}.
    data_logger_bank.register(APP_CLIENT_KEY, app_client_data_logger)
    # performance recorder -------------------------------------------------------------------------
    app_client_ms_performance_recorder = MSPerformanceRecorder(
        descr=APP_CLIENT_KEY, data_logger=app_client_data_logger,
        fields_to_report=['p99', 'avg_latency'])
    ms_performance_recorder_bank.register(APP_CLIENT_KEY, app_client_ms_performance_recorder)
    # load forecaster ------------------------------------------------------------------------------
    app_client_load_forecaster = TSBaseLearner(
        app_id=APP_CLIENT_KEY, data_logger=app_client_data_logger, model='arima-default',
        field_to_forecast='load')
    load_forecaster_bank.register(APP_CLIENT_KEY, app_client_load_forecaster)
    app_client_load_forecaster.initialise()
    # model and learner ----------------------------------------------------------------------------
    if args.policy in ['ucbopt']:
        app_client_model = GP(APP_CLIENT_KEY, APP_CLIENT_KEY, alloc_leaf_order)
        app_client_learner = P99Learner(app_id=APP_CLIENT_KEY, data_logger=app_client_data_logger,
                                        model=app_client_model)
        learner_bank.register(APP_CLIENT_KEY, app_client_learner)
        app_client_learner.initialise()
    # Create policy ================================================================================
    if args.policy == 'propfair':
        policy = PropFairness(env=env, resource_quantity=framework_manager.get_cluster_resources(),
                              load_forecaster_bank=load_forecaster_bank,
                              alloc_granularity=framework_manager.get_alloc_granularity())
    elif args.policy == 'ucbopt':
        policy = UCBOpt(env=env, resource_quantity=framework_manager.get_cluster_resources(),
                        learner_bank=learner_bank, load_forecaster_bank=load_forecaster_bank,
                        app_client_key=APP_CLIENT_KEY,
                        num_iters_for_evo_opt=args.num_iters_for_evo_opt)
    elif args.policy == 'msevoopt':
        policy = MSEvoOpt(env=env, resource_quantity=framework_manager.get_cluster_resources(),
                          load_forecaster_bank=load_forecaster_bank,
                          data_logger_bank=data_logger_bank,
                          field_to_minimise='p99', app_client_key=APP_CLIENT_KEY)
    elif args.policy == 'msile':
        policy = MSInterleavedExploration(
            env=env, resource_quantity=framework_manager.get_cluster_resources(),
            load_forecaster_bank=load_forecaster_bank,
            data_logger_bank=data_logger_bank,
            field_to_minimise='p99', app_client_key=APP_CLIENT_KEY)
    else:
        raise ValueError('Unknown policy_name %s.' % args.policy)
    policy.initialise()
    logger.info('Initialised policy %s.', str(policy))

    # Pass learner bank and time series model to the scheduler =====================================
    cilantro = CilantroScheduler(event_queue=event_queue,
                                 framework_manager=framework_manager,
                                 event_logger=event_logger,
                                 env=env,
                                 policy=policy,
                                 data_logger_bank=data_logger_bank,
                                 learner_bank=learner_bank,
                                 performance_recorder_bank=ms_performance_recorder_bank,
                                 load_forecaster_bank=load_forecaster_bank,
                                 learner_datapoll_frequency=LEARNER_DATAPOLL_FREQUENCY)
    # Initiate learning/reporting etc --------------------------------------------------------------
    ms_performance_recorder_bank.initiate_report_results_loop()
    data_logger_bank.initiate_write_to_disk_loop()
    learner_bank.initiate_training_loop()
    load_forecaster_bank.initiate_training_loop()
    reward_forecaster_bank.initiate_training_loop()

    # Create the workloads and deploy them =========================================================
    # Deploy by creating all YAMLs in the workloads directory
    all_deps_ready = False
    while not all_deps_ready:
        current_deps = framework_manager.get_deployments().keys()
        all_deps_ready = all(('root--' + d in current_deps) for d in HOTELRES_MICROSERVICES)
        if not all_deps_ready:
            not_ready_deps = [d for d in HOTELRES_MICROSERVICES if
                              ('root--' + d not in current_deps)]
            ready_deps = [d for d in HOTELRES_MICROSERVICES if
                              ('root--' + d in current_deps)]
            logger.info(f"Not all microservices are ready.\nNot ready: {not_ready_deps}.\n" +
                        f"Ready: {ready_deps}.\nUse kubectl create -Rf <path_to_yamls> to run them."
                        f" Currently running: {list(current_deps)}.")
            time.sleep(1)
    logger.info('Workloads ready!')
    # Create event sources -------------------------------------------------------------------------
    for s in event_sources:
        event_loop.create_task(s.event_generator())
    try:
        event_loop.run_until_complete(cilantro.scheduler_loop())
    finally:
        event_loop.close()


if __name__ == '__main__':
    main()
