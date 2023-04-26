"""
    Loads a cilantro scheduler by first loading profiled data and then running the policy.
    -- romilbhardwaj
    -- kirthevasank
"""

# pylint: disable=too-many-locals
# pylint: disable=too-many-statements
# pylint: disable=too-many-branches

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
from cilantro.core.performance_recorder import PerformanceRecorder, PerformanceRecorderBank
from cilantro.data_loggers.data_logger_bank import DataLoggerBank
from cilantro.data_loggers.simple_data_logger import SimpleDataLogger
from cilantro.data_loggers.simple_event_logger import SimpleEventLogger
from cilantro.learners.base_learner import BaseLearner
from cilantro.learners.ibtree import IntervalBinaryTree
from cilantro.learners.timennls import TimeNNLS
from cilantro.learners.learner_bank import LearnerBank
from cilantro.policies.autoscaling import BanditAutoScaler, OracularAutoScaler
from cilantro.policies.as_baselines import K8sAutoScaler, PIDAutoScaler, DS2AutoScaler
from cilantro.policies.ernest import Ernest
from cilantro.policies.prop_fairness import PropFairness
from cilantro.policies.minerva import Minerva
from cilantro.policies.mmflearn import MMFLearn
from cilantro.policies.mmf import HMMFDirect
from cilantro.policies.multincadddec import MultIncAddDec
from cilantro.policies.parties import Parties
from cilantro.policies.quasar import Quasar
from cilantro.policies.welfare_policy import UtilWelfareBanditPolicy, UtilWelfareOracularPolicy, \
                                             EgalWelfareBanditPolicy, EgalWelfareOracularPolicy
from cilantro.policies.maximin import EgalWelfareGreedy
from cilantro.policies.evo_alg_welfare import UtilWelfareEvoAlg, EgalWelfareEvoAlg
from cilantro.profiling.profiled_info_loader import ProfiledInfoBank
from cilantro.scheduler.cilantroscheduler import CilantroScheduler
from cilantro.timeseries.ts_forecaster_bank import TSForecasterBank
from cilantro.timeseries.ts_base_learner import TSBaseLearner
from cilantro.workloads.k8s_workload_deployer import K8sWorkloadDeployer
# In Demo
from env_gen import generate_env

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s | %(levelname)-6s | %(name)-40s || %(message)s',
                    datefmt='%m-%d %H:%M:%S'
                    )
logger = logging.getLogger(__name__)

DFLT_REAL_OR_DUMMY = 'dummy'
# DFLT_REAL_OR_DUMMY = 'real'

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
NUM_PARALLEL_TRAINING_THREADS = 8
SLEEP_TIME_BETWEEN_TRAINING = 5

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
                        help='Environment for running experiment.')
    parser.add_argument('--cluster-type', '-clus', type=str,
                        help='Which cluster_type to rund, eks or kind.')
    parser.add_argument('--real-or-dummy', '-rod', type=str, default=DFLT_REAL_OR_DUMMY,
                        help='To run a real or dummy workload.')
    parser.add_argument('--profiled-info-dir', '-pid', type=str,
                        help='Directory which has the profiled data saved.')
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
    logger.info('Created Env: %s.\n%s', str(env), env.write_to_file(None))
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
            workload_type=leaf.get_workload_info('workload_type'))
        data_logger_bank.register(leaf_path, data_logger)
        # Create the performance recorder ---------------------------------------------------
        leaf_unit_demand = profiled_info_bank.get_unit_demand_for_leaf_node(leaf)
                # N.B: leaf_unit_demand requires oracular information and in our real experiments,
                # will come from profiling information.
        performance_recorder = PerformanceRecorder(
            app_id=leaf_path, performance_goal=leaf.threshold, unit_demand=leaf_unit_demand,
            util_scaling=leaf.util_scaling, entitlement=entitlements[leaf_path],
            data_logger=data_logger)
        performance_recorder_bank.register(leaf_path, performance_recorder)
        # Create the time series learner ----------------------------------------------------
        if not (args.policy in ['propfair']):
            load_forecaster = TSBaseLearner(
                app_id=leaf_path, data_logger=data_logger, model='arima-default',
                field_to_forecast='load')
            load_forecaster_bank.register(leaf_path, load_forecaster)
            load_forecaster.initialise()
        # Create the model and learner ------------------------------------------------------
        if args.policy in ['mmflearn', 'utilwelflearn', 'egalwelflearn', 'aslearn']:
            lip_const = profiled_info_bank.get_info_for_leaf_node(leaf, 'lip_const')
            int_ub = profiled_info_bank.get_info_for_leaf_node(leaf, 'int_ub')
            model = IntervalBinaryTree(leaf_path, int_lb=0, int_ub=int_ub,
                                       lip_const=lip_const) # Can customise for each leaf.
            learner = BaseLearner(
                app_id=leaf_path, data_logger=data_logger, model=model)
            learner_bank.register(leaf_path, learner)
            learner.initialise()
        elif args.policy == 'ernest':
            int_ub = profiled_info_bank.get_info_for_leaf_node(leaf, 'int_ub')
            model = TimeNNLS(leaf_path, int_lb=0, int_ub=int_ub)
            learner = BaseLearner(app_id=leaf_path, data_logger=data_logger, model=model)
            learner_bank.register(leaf_path, learner)
            learner.initialise()
        elif args.policy == 'minerva':
            reward_forecaster = TSBaseLearner(
                app_id=leaf_path, data_logger=data_logger, model='arima-default',
                field_to_forecast='reward')
            reward_forecaster_bank.register(leaf_path, reward_forecaster)
            reward_forecaster.initialise()
        elif args.policy == 'mmf':
            leaf.set_unit_demand(leaf_unit_demand)

    # Create policy ================================================================================
    if args.policy == 'propfair':
        policy = PropFairness(env=env, resource_quantity=framework_manager.get_cluster_resources(),
                              load_forecaster_bank=load_forecaster_bank,
                              alloc_granularity=framework_manager.get_alloc_granularity())
    # Fairness policies 00--------------------------------------------------------------------------
    elif args.policy == 'mmf':
        policy = HMMFDirect(env=env, resource_quantity=framework_manager.get_cluster_resources(),
                            load_forecaster_bank=load_forecaster_bank,
                            alloc_granularity=framework_manager.get_alloc_granularity())
    elif args.policy == 'mmflearn':
        policy = MMFLearn(env=env, resource_quantity=framework_manager.get_cluster_resources(),
                          load_forecaster_bank=load_forecaster_bank, learner_bank=learner_bank,
                          alloc_granularity=framework_manager.get_alloc_granularity())
    # Utilitarian welfare --------------------------------------------------------------------------
    elif args.policy == 'utilwelforacle':
        policy = UtilWelfareOracularPolicy(
            env=env, resource_quantity=framework_manager.get_cluster_resources(),
            load_forecaster_bank=load_forecaster_bank, profiled_info_bank=profiled_info_bank,
            alloc_granularity=framework_manager.get_alloc_granularity(),
            num_iters_for_evo_opt=args.num_iters_for_evo_opt)
    elif args.policy == 'utilwelflearn':
        policy = UtilWelfareBanditPolicy(
            env=env, resource_quantity=framework_manager.get_cluster_resources(),
            load_forecaster_bank=load_forecaster_bank, learner_bank=learner_bank,
            alloc_granularity=framework_manager.get_alloc_granularity(),
            num_iters_for_evo_opt=args.num_iters_for_evo_opt)
    elif args.policy == 'evoutil':
        policy = UtilWelfareEvoAlg(
            env=env, resource_quantity=framework_manager.get_cluster_resources(),
            performance_recorder_bank=performance_recorder_bank,
            load_forecaster_bank=load_forecaster_bank,
            alloc_granularity=framework_manager.get_alloc_granularity())
    # Egalitarian welfare --------------------------------------------------------------------------
    elif args.policy == 'egalwelforacle':
        policy = EgalWelfareOracularPolicy(
            env=env, resource_quantity=framework_manager.get_cluster_resources(),
            load_forecaster_bank=load_forecaster_bank, profiled_info_bank=profiled_info_bank,
            alloc_granularity=framework_manager.get_alloc_granularity(),
            num_iters_for_evo_opt=args.num_iters_for_evo_opt)
    elif args.policy == 'egalwelflearn':
        policy = EgalWelfareBanditPolicy(
            env=env, resource_quantity=framework_manager.get_cluster_resources(),
            load_forecaster_bank=load_forecaster_bank, learner_bank=learner_bank,
            alloc_granularity=framework_manager.get_alloc_granularity(),
            num_iters_for_evo_opt=args.num_iters_for_evo_opt)
    elif args.policy == 'evoegal':
        policy = EgalWelfareEvoAlg(
            env=env, resource_quantity=framework_manager.get_cluster_resources(),
            performance_recorder_bank=performance_recorder_bank,
            load_forecaster_bank=load_forecaster_bank,
            alloc_granularity=framework_manager.get_alloc_granularity())
    elif args.policy == 'greedyegal':
        policy = EgalWelfareGreedy(
            env=env, resource_quantity=framework_manager.get_cluster_resources(),
            performance_recorder_bank=performance_recorder_bank,
            load_forecaster_bank=load_forecaster_bank,
            alloc_granularity=framework_manager.get_alloc_granularity())
    # Other policies -------------------------------------------------------------------------------
    elif args.policy == 'minerva':
        policy = Minerva(env=env, resource_quantity=framework_manager.get_cluster_resources(),
                         load_forecaster_bank=load_forecaster_bank,
                         reward_forecaster_bank=reward_forecaster_bank,
                         alloc_granularity=framework_manager.get_alloc_granularity())
    elif args.policy == 'ernest':
        policy = Ernest(env, resource_quantity=framework_manager.get_cluster_resources(),
                        load_forecaster_bank=load_forecaster_bank, learner_bank=learner_bank,
                        alloc_granularity=framework_manager.get_alloc_granularity())
    elif args.policy == 'quasar':
        policy = Quasar(env, resource_quantity=framework_manager.get_cluster_resources(),
                        performance_recorder_bank=performance_recorder_bank,
                        load_forecaster_bank=load_forecaster_bank,
                        alloc_granularity=framework_manager.get_alloc_granularity())
    elif args.policy == 'parties':
        policy = Parties(env, resource_quantity=framework_manager.get_cluster_resources(),
                         load_forecaster_bank=load_forecaster_bank,
                         reward_forecaster_bank=reward_forecaster_bank,
                         alloc_granularity=framework_manager.get_alloc_granularity())
    elif args.policy == 'multincadddec':
        policy = MultIncAddDec(env, resource_quantity=framework_manager.get_cluster_resources(),
                               load_forecaster_bank=load_forecaster_bank,
                               reward_forecaster_bank=reward_forecaster_bank,
                               alloc_granularity=framework_manager.get_alloc_granularity())
    # Autoscaling policies -------------------------------------------------------------------------
    elif args.policy == 'asoracle':
        policy = OracularAutoScaler(
            env=env, resource_quantity=framework_manager.get_cluster_resources(),
            load_forecaster_bank=load_forecaster_bank,
            profiled_info_bank=profiled_info_bank)
    elif args.policy == 'aslearn':
        policy = BanditAutoScaler(
            env=env, resource_quantity=framework_manager.get_cluster_resources(),
            load_forecaster_bank=load_forecaster_bank,
            learner_bank=learner_bank)
    elif args.policy == 'k8sas':
        policy = K8sAutoScaler(
            env=env, resource_quantity=framework_manager.get_cluster_resources(),
            performance_recorder_bank=performance_recorder_bank)
    elif args.policy == 'pidas':
        policy = PIDAutoScaler(
            env=env, resource_quantity=framework_manager.get_cluster_resources(),
            performance_recorder_bank=performance_recorder_bank)
    elif args.policy == 'ds2':
        policy = DS2AutoScaler(
            env=env, resource_quantity=framework_manager.get_cluster_resources(),
            load_forecaster_bank=load_forecaster_bank,
            performance_recorder_bank=performance_recorder_bank)
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
    # Initiate learning/reporting etc --------------------------------------------------------------
    performance_recorder_bank.initiate_report_results_loop()
    data_logger_bank.initiate_write_to_disk_loop()
    learner_bank.initiate_training_loop()
    load_forecaster_bank.initiate_training_loop()
    reward_forecaster_bank.initiate_training_loop()

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

