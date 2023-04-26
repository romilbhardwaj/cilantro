"""
    Harness for processing events provided by an event queue.
    -- romilbhardwaj
    -- kirthevasank
"""

import asyncio
import logging
import os
import time
from typing import Union

from cilantro.backends.base_framework_manager import BaseFrameworkManager
from cilantro.core.henv import TreeEnvironment
from cilantro.core.performance_recorder import PerformanceRecorderBank, PerformanceRecorder
from cilantro.data_loggers.data_logger_bank import DataLoggerBank
from cilantro.data_loggers.simple_data_logger import SimpleDataLogger
from cilantro.learners.base_learner import BaseLearner
from cilantro.learners.ibtree import IntervalBinaryTree
from cilantro.learners.learner_bank import LearnerBank
from cilantro.policies.base_policy import BasePolicy
from cilantro.policies.mmflearn import MMFLearn
from cilantro.policies.prop_fairness import PropFairness
from cilantro.timeseries.ts_base_learner import TSBaseLearner
from cilantro.timeseries.ts_forecaster_bank import TSForecasterBank
from cilantro.types.events import AllocExpirationEvent, AppAddEvent, AppUpdateEvent, BaseEvent, \
                                  EventTypes, UtilityUpdateEvent

logger = logging.getLogger(__name__)

DEFAULT_LEARNER_THREADS = 10
DATALOGGER_FIELDS = ['load', 'alloc', 'reward', 'sigma', 'event_start_time', 'event_end_time']
LIP_CONST = 1

# Set debug flag if environment variable is set
PERF_DEBUG = True if 'CILANTRO_PERF_DEBUG' in os.environ else False

class CilantroScheduler:
    """ Cilantro scheduler. """

    def __init__(self,
                 event_queue: asyncio.Queue,
                 framework_manager: BaseFrameworkManager,
                 event_logger: object,
                 env: object = None,
                 policy: Union[str, BasePolicy] = None,
                 data_logger_bank: object = None,
                 learner_bank: object = None,
                 load_forecaster_bank: object = None,
                 add_learners_if_not_given: bool = False,
                 performance_recorder_bank: object = None,
                 learner_datapoll_frequency: float = 1):
        """
        Main cilantro scheduler class. Responsible for processing events in the provided event queue
        The cilantro scheduler also reads the environment (dummy/kubernetes) and creates
        the env and policy objects in the __init__ function.
        :param event_queue: Main event queue
        :param event_logger: Event logger used for debug and storing events in the system.
        :param policy: String identifying the policy to be used or Policy object. See get_policy
                       method.
        :param framework_manager: Framework manager (k8s or default)
        :param learner_datapoll_frequency: How frequently (in seconds) data is polled by TSLearner
               and BaseLearner.
        """
        # Scheduler objects
        self.event_queue = event_queue
        self.event_logger = event_logger
        self.framework_manager = framework_manager
        # Other parameters
        self.total_res = self.framework_manager.get_cluster_resources()
        self.alloc_granularity = self.framework_manager.get_alloc_granularity()
        # Get asyncio loop
        self.aioloop = asyncio.get_event_loop()
        # Init env and policy
        self.env = env if env else self._get_init_env()
        self.policy = self._get_policy(policy) if isinstance(policy, str) else policy
        self.data_logger_bank = data_logger_bank if data_logger_bank else DataLoggerBank()
        self.learner_bank = learner_bank if learner_bank else LearnerBank()
        self.load_forecaster_bank = load_forecaster_bank if load_forecaster_bank \
                                    else TSForecasterBank()
        self.performance_recorder_bank = \
            performance_recorder_bank if performance_recorder_bank else \
            PerformanceRecorderBank(resource_quantity=self.total_res,
                                    alloc_granularity=self.alloc_granularity)
        self.learner_datapoll_frequency = learner_datapoll_frequency
        self.add_learners_if_not_given = add_learners_if_not_given

    @classmethod
    def _get_init_env(cls):
        """ Obtain initial environment. """
        env = TreeEnvironment(None, 1)
        return env

    def _get_policy(self, policy_str: str):
        """ Obtain policy from string. """
        if policy_str == 'propfair':
            policy = PropFairness(env=self.env, resource_quantity=self.total_res,
                                  alloc_granularity=self.alloc_granularity)
        elif policy_str == 'mmflearn':
            policy = MMFLearn(env=self.env, learner_bank=self.learner_bank,
                              load_forecaster_bank=self.load_forecaster_bank,
                              resource_quantity=self.total_res,
                              alloc_granularity=self.alloc_granularity)
        else:
            raise ValueError('Unknown policy name %s.' % (policy_str))
        policy.initialise()
        logger.debug('Getting policy %s from string %s.', str(policy), policy_str)
        return policy

    def process_event(self, event: BaseEvent):
        '''
        Sends an event to the appropriate processor.
        :param event:
        :return:
        '''
        if event.event_type == EventTypes.UTILITY_UPDATE:
            assert isinstance(event, UtilityUpdateEvent)
            self.processor_utility_event(event)
        elif event.event_type == EventTypes.APP_ADDED:
            assert isinstance(event, AppAddEvent)
            self.processor_app_add_event(event)
        elif event.event_type == EventTypes.APP_REMOVED:
            assert isinstance(event, AppUpdateEvent)
            self.processor_app_remove_event(event)
        elif event.event_type == EventTypes.ALLOC_TIMEOUT:
            assert isinstance(event, AllocExpirationEvent)
            self.processor_alloc_expiration_event(event)
        else:
            raise NotImplementedError(f"Event type {event.event_type} is not supported.")


    def processor_alloc_expiration_event(self, event: AllocExpirationEvent):
        """ Process an allocation expiration event. """
        # 1) Invoke policy to get resource allocation
        start_time = time.time()
        new_allocation = self.policy.get_resource_allocation()  # dict {app_path: allocation}
        total_time = time.time() - start_time
        logger.debug("Received new allocation from policy - %s", str(new_allocation))
        if PERF_DEBUG:
            logger.debug("Time taken to compute allocation - {:.4f}".format(total_time))
            # Append to time taken to file
            with open('/tmp/cilantro_perf.txt', 'a') as f:
                f.write(f'{len(self.env.leaf_nodes)},{str(total_time)}\n')
        # 2) Execute resource allocation
        self.framework_manager.apply_allocation(new_allocation)
        logger.debug("Executed resource allocation from framework manager.")
        # 3) Inform the event source.
        event.inform_event_source_of_allocation()
        # N.B (@Romil): we probably don't want to execute a resource allocation every time there is
        # a utility update (since utility updates happen every few seconds). The expiration event
        # (which we can call every few minutes or so depending on how agile k8s is will execute the
        # resource allocation at a slower rate than utility updates. We might want to consider also
        # including a significant-load-change event to trigger resource changes.


    def processor_app_add_event(self, event: AppAddEvent):
        """ Process an add app event. """
        # On app add, we need to create new learners, update env and ....
        self._add_app_to_env(event.app_path, event.app_threshold, event.app_weight,
                             event.app_unit_demand)

    def _add_app_to_env(self, app_path, app_threshold, app_weight, app_unit_demand):
        """ Add app to the environment. """
        # Performs all operations required by MMFL structures (env, policy, learner bank) on app add
        # Create leaf node in env
        try:
            node_added_to_tree = self.env.add_nodes_to_tree_from_path(
                app_path, app_weight, app_threshold, leaf_unit_demand=app_unit_demand,
                last_node_is_a_leaf_node=True, update_tree_at_end=True)
        except KeyError as e:
            logger.error(f"CRITICAL: App {app_path} failed to add to tree. Not retrying. f{str(e)}")
        leaf_entitlements = self.env.get_entitlements()

        # Create datalogger ------------------------------------------------------------------
        if not (self.data_logger_bank.tag_exists(app_path)):
            logger.info('Adding %s to data logger bank in CilantroScheduler.', app_path)
            data_logger = SimpleDataLogger(app_path, DATALOGGER_FIELDS,
                                           index_fld='event_start_time')
            self.data_logger_bank.register(app_path, data_logger)
        else:
            data_logger = self.data_logger_bank.get(app_path)
#         # Create performance recorder and register it in the bank ----------------------------
#         if not (self.performance_recorder_bank.tag_exists(app_path)):
#             logger.info('Adding %s to performance recorder bank in CilantroScheduler.', app_path)
#             app_entitlement = leaf_entitlements[app_path]
#             #TODO(romilb): util_scaling must be propagated from upstream methods..
#             #  typically the performance recorder is populated directly in env so not needed.
#             performance_recorder = PerformanceRecorder(app_id=app_path,
#                                                        performance_goal=app_threshold,
#                                                        util_scaling='linear',
#                                                        unit_demand=app_unit_demand,
#                                                        entitlement=app_entitlement,
#                                                        data_logger=data_logger)
#             self.performance_recorder_bank.register(app_path, performance_recorder)
        # Create the load forecaster, register it and initialize ------------------------------
        if self.add_learners_if_not_given and \
            not (self.load_forecaster_bank.tag_exists(app_path)):
            logger.info('Adding %s to load forecaster bank in CilantroScheduler.', app_path)
            load_forecaster = TSBaseLearner(
                app_id=app_path, data_logger=data_logger,
                model='arima-default',
                field_to_forecast='load')
            self.load_forecaster_bank.register(app_path, load_forecaster)
            load_forecaster.initialise()
        # Create model and learner ------------------------------------------------------------
        if self.add_learners_if_not_given and \
            not (self.learner_bank.tag_exists(app_path)):
            logger.info('Adding %s to learner bank in CilantroScheduler.', app_path)
            model = IntervalBinaryTree(app_path, int_lb=0, int_ub=self.total_res,
                                       lip_const=LIP_CONST)
            learner = BaseLearner(app_id=app_path, data_logger=data_logger, model=model)
            self.learner_bank.register(app_path, learner)
            learner.initialise()
        # If node was added to tree, change all entitlements ----------------------------------
        if node_added_to_tree:
            for tag, performance_recorder in self.performance_recorder_bank.enumerate():
                performance_recorder.set_entitlement(leaf_entitlements[tag])

    def processor_app_remove_event(self, event: AppUpdateEvent):
        """ App remove event. """
        # On app add, we need to remove learners, update policy and ....
        raise NotImplementedError

    def processor_utility_event(self, event: UtilityUpdateEvent):
        """ Process utility event. """
        # 1a) Log utility updates. Get the appropriate data logger and update.
        data_logger = self.data_logger_bank.get(tag=event.app_path)
        data_logger.log_event(event.__dict__())
        # Note that learners pull from the data logger in a separate thread,
        # so no learning work needs to be done here.

    async def scheduler_loop(self):
        '''
        :param event_generators:
        :return:
        '''
        while True:
            logger.debug("Waiting for event.")

            # Fetch event
            event = await self.event_queue.get()
            logger.debug(str(event))
            self.event_logger.log_event(event)

            # Parse and handle event
            self.process_event(event)
