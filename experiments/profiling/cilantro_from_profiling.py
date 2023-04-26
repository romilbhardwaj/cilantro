"""
    Loads a cilantro scheduler by first loading profiled data and then running the policy.
    -- kirthevasank
"""

# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=too-many-branches
# pylint: disable=import-error

import argparse
import asyncio
import os
from datetime import datetime
import logging
# Local
from cilantro.ancillary.info_write_load_utils import write_experiment_info_to_files
from cilantro.backends.alloc_expiration_event_source import AllocExpirationEventSource
from cilantro.backends.grpc.utility_event_source import UtilityEventSource
from cilantro.backends.k8s.kubernetes_manager import KubernetesManager
from cilantro.backends.test.test_backend import DummyFrameworkManager, DummyEventSource
from cilantro.core.performance_recorder import PerformanceRecorder, PerformanceRecorderBank
from cilantro.data_loggers.data_logger_bank import DataLoggerBank
from cilantro.data_loggers.simple_data_logger import SimpleDataLogger
from cilantro.data_loggers.simple_event_logger import SimpleEventLogger
from cilantro.learners.base_learner import BaseLearner
from cilantro.learners.ibtree import IntervalBinaryTree
from cilantro.learners.learner_bank import LearnerBank
from cilantro.policies.prop_fairness import PropFairness
from cilantro.policies.minerva import Minerva
from cilantro.policies.mmflearn import MMFLearn
from cilantro.policies.mmf import HMMFDirect
from cilantro.profiling.profiled_info_loader import ProfiledInfoBank
from cilantro.scheduler.cilantroscheduler import CilantroScheduler
from cilantro.timeseries.ts_forecaster_bank import TSForecasterBank
from cilantro.timeseries.ts_base_learner import TSBaseLearner
from cilantro.workloads.k8s_workload_deployer import K8sWorkloadDeployer
# In Demo
from env_demo import generate_env

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)-6s | %(name)-40s || %(message)s',
                    datefmt='%m-%d %H:%M:%S'
                    )
# logger = logging.getLogger(__name__)

# POLICY_NAME = 'propfair'
# POLICY_NAME = 'mmflearn'
# POLICY_NAME = 'minerva'
# POLICY_NAME = 'mmf'

WRITE_LOG_TO_FILE = True
# WRITE_LOG_TO_FILE = False

DUMMY_NUM_RESOURCES = 60

ENV_DESCR = 'simple'
INT_UPPER_BOUND = 25
LOAD_FILE = None

# ENV_DESCR = 'twitter_1476'
# INT_UPPER_BOUND = 0.03
# LOAD_FILE = 'twitter_1476_data'

ALLOC_GRANULARITY = 1 # we cannot assign fractional resources

# For the data logger -----------------------------------------------------
MAX_INMEM_TABLE_SIZE = -1
MAX_INMEM_TABLE_SIZE = 1000
DATA_LOG_WRITE_TO_DISK_EVERY = 30

# Other parameters
ASYNC_SLEEP_TIME = 1
SLEEP_TIME_BETWEEN_DATA_REPOLLS = 1.1
ALLOC_EXPIR_TIME = 6 # Allocate every this many seconds
GRPC_PORT = 10000
LEARNER_DATAPOLL_FREQUENCY = 5  # Learners fetch from data loggers every these many seconds

# For logging and saving results ----------------------------------------
SCRIPT_TIME_STR = datetime.now().strftime('%m%d%H%M%S')
# log_file_name = 'logs/%s_%s_%s.log'%(POLICY_NAME, ENV_DESCR, SCRIPT_TIME_STR)
# logging.basicConfig(filename=log_file_name, level=logging.INFO)
# logger = logging.getLogger(__name__)
REPORT_RESULTS_EVERY = 17

DEBUG_MODE = False


def main():
    """ Main function. """
    # Parse args and create directories ============================================================
    parser = argparse.ArgumentParser(description='Arguments for running demo_async.py.')
    parser.add_argument('--profiled-info-dir', '-pid', type=str,
                        help='Directory which has the profiled data saved.')
    parser.add_argument('--policy', '-pol', type=str,
                        help='Which policy to run.')
    parser.add_argument('--cluster-type', '-clus', type=str,
                        help='Which cluster_type to rund, eks or kind.')
    args = parser.parse_args()

    # Create the environment and other initial set up ==============================================
    env = generate_env(args.cluster_type)
    entitlements = env.get_entitlements()
    logging.info('Created Env: %s,\n%s\n', str(env), env.write_to_file(None))
    # Create directories for logging results
    profiled_info_bank = ProfiledInfoBank(args.profiled_info_dir)
    logging.info('Loading profiled information from %s.', args.profiled_info_dir)
    logging.info(profiled_info_bank.display_profiled_information_for_env(env))

    # Create event loggers, framework managers, and event sources ==================================
    event_queue = asyncio.Queue()
    event_logger = SimpleEventLogger()
    event_loop = asyncio.get_event_loop()
    if not DEBUG_MODE:
        framework_manager = KubernetesManager(event_queue,
                                              update_loop_sleep_time=1,
                                              dry_run=False)
        event_sources = [UtilityEventSource(output_queue=event_queue, server_port=GRPC_PORT)]
    else:
        env_jobs = ["root--c1--j1"]
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
    experiment_workdir = 'workdirs/%s_%s_%d_%s_%s'%(args.policy, ENV_DESCR, num_resources,
                                                    args.cluster_type, SCRIPT_TIME_STR)
    if not os.path.exists(experiment_workdir):
        os.makedirs(experiment_workdir, exist_ok=True)
    save_results_file_path = os.path.join(experiment_workdir, 'in_run_results.p')
    # Write experimental information to file before commencing experiments ------------------------
    experiment_info = {}
    experiment_info['resource_quantity'] = num_resources
    experiment_info['alloc_granularity'] = framework_manager.get_alloc_granularity()
    write_experiment_info_to_files(experiment_workdir, env, experiment_info)
    if WRITE_LOG_TO_FILE:
        log_file_name = os.path.join(experiment_workdir, 'log.log')
        logging.basicConfig(filename=log_file_name, level=logging.INFO,
                            format='%(asctime)s | %(levelname)-6s | %(name)-40s || %(message)s',
                            datefmt='%m-%d %H:%M:%S'
                            )
        logger = logging.getLogger(__name__)
        logger.info('Created Env: %s,\n%s', str(env), env.write_to_file(None))
        logger.info('Loading profiled information from %s.', args.profiled_info_dir)
    else:
        logger = logging.getLogger(__name__)

    # Create data loggers, learners, time_series forcaseters and performance recorder for each leaf
    # node ========================================================================================
    load_forecaster_bank = TSForecasterBank()
    data_logger_bank = DataLoggerBank()
    learner_bank = LearnerBank()
    reward_forecaster_bank = TSForecasterBank()
    performance_recorder_bank = PerformanceRecorderBank(
        resource_quantity=framework_manager.get_cluster_resources(),
        alloc_granularity=framework_manager.get_alloc_granularity(),
        report_results_every=REPORT_RESULTS_EVERY, save_file_name=save_results_file_path,
        report_results_descr=args.policy,
        )
    for leaf_path, leaf in env.leaf_nodes.items():
        # Create the data logger ------------------------------------------------------------
        # In practice, we might have to compute sigma from other raw metrics.
        data_logger = SimpleDataLogger(
            leaf_path, ['load', 'alloc', 'reward', 'sigma', 'event_start_time', 'event_end_time'],
            index_fld='event_start_time', max_inmem_table_size=MAX_INMEM_TABLE_SIZE,
            workload_type=leaf.get_workload_info('workload_type'),
            disk_dir=experiment_workdir, write_to_disk_every=DATA_LOG_WRITE_TO_DISK_EVERY)
        data_logger_bank.register(leaf_path, data_logger)
        # Create the performance recorder ---------------------------------------------------
        leaf_unit_demand = profiled_info_bank.get_unit_demand_for_leaf_node(leaf)
                # N.B: leaf_unit_demand requires oracular information and in our real experiments,
                # will come from profiling information.
        performance_recorder = PerformanceRecorder(
            app_id=leaf_path, performance_goal=leaf.threshold, unit_demand=leaf_unit_demand,
            entitlement=entitlements[leaf_path], data_logger=data_logger)
        performance_recorder_bank.register(leaf_path, performance_recorder)
        # Create the time series learner ----------------------------------------------------
        # Create the time series learner ----------------------------------------------------
        if args.policy in ['mmf', 'mmflearn', 'minerva']:
            load_forecaster = TSBaseLearner(
                app_id=leaf_path, data_logger=data_logger, model='arima-default',
                field_to_forecast='load',
                sleep_time_between_data_repolls=SLEEP_TIME_BETWEEN_DATA_REPOLLS)
            load_forecaster_bank.register(leaf_path, load_forecaster)
            load_forecaster.initialise()
        # Create the model and learner ------------------------------------------------------
        if args.policy == 'mmflearn':
            lip_const = profiled_info_bank.get_info_for_leaf_node(leaf, 'lip_const')
            model = IntervalBinaryTree(leaf_path, int_lb=0, int_ub=INT_UPPER_BOUND,
                                       lip_const=lip_const) # Can customise for each leaf.
            learner = BaseLearner(
                app_id=leaf_path, data_logger=data_logger, model=model,
                sleep_time_between_data_repolls=SLEEP_TIME_BETWEEN_DATA_REPOLLS)
            learner_bank.register(leaf_path, learner)
            learner.initialise()
        elif args.policy == 'minerva':
            reward_forecaster = TSBaseLearner(
                app_id=leaf_path, data_logger=data_logger, model='arima-default',
                field_to_forecast='reward',
                sleep_time_between_data_repolls=SLEEP_TIME_BETWEEN_DATA_REPOLLS)
            reward_forecaster_bank.register(leaf_path, reward_forecaster)
            reward_forecaster.initialise()
        elif args.policy == 'mmf':
            leaf.set_unit_demand(leaf_unit_demand)

    # Create policy ================================================================================
    if args.policy == 'propfair':
        policy = PropFairness(env=env, resource_quantity=framework_manager.get_cluster_resources(),
                              load_forecaster_bank=load_forecaster_bank,
                              alloc_granularity=framework_manager.get_alloc_granularity())
    elif args.policy == 'mmflearn':
        policy = MMFLearn(env=env, resource_quantity=framework_manager.get_cluster_resources(),
                          load_forecaster_bank=load_forecaster_bank, learner_bank=learner_bank,
                          alloc_granularity=framework_manager.get_alloc_granularity())
    elif args.policy == 'minerva':
        policy = Minerva(env=env, resource_quantity=framework_manager.get_cluster_resources(),
                         load_forecaster_bank=load_forecaster_bank,
                         reward_forecaster_bank=reward_forecaster_bank, alloc_granularity=1)
    elif args.policy == 'mmf':
        policy = HMMFDirect(env=env, resource_quantity=framework_manager.get_cluster_resources(),
                            load_forecaster_bank=load_forecaster_bank, alloc_granularity=1)
    else:
        raise ValueError('Unknown policy_name %s.'%(args.policy))
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
                                 performance_recorder_bank=performance_recorder_bank,
                                 load_forecaster_bank=load_forecaster_bank,
                                 learner_datapoll_frequency=LEARNER_DATAPOLL_FREQUENCY)
    # Initiate reporting of results ----------------------------------------------------------------
    performance_recorder_bank.initiate_report_results_loop()

    # Create the workloads and deploy them =========================================================
    workload_exec = K8sWorkloadDeployer()
    workload_exec.deploy_environment(env)
    logger.info('Workload deployed!')
    # Create event sources -------------------------------------------------------------------------
    for s in event_sources:
        event_loop.create_task(s.event_generator())
    try:
        event_loop.run_until_complete(cilantro.scheduler_loop())
    finally:
        event_loop.close()


if __name__ == '__main__':
    main()

