"""
    A base learner class.
    -- romilbhardwaj
    -- kirthevasank
"""

# pylint: disable=logging-not-lazy

import logging
from copy import deepcopy
# local
from cilantro.learners.learner_bank import DemandRecommender

# How the learner works: The learner polls the data logger for new data. If it receives new
# data, it processes the data and updates the model and keeps repeating this infinitely.

DFLT_SLEEP_TIME_BETWEEN_DATA_REPOLLS = 10
NUM_INIT_DATA_TO_IGNORE = 5

logger = logging.getLogger(__name__)


class LearningModel:
    """ Learning Model. """

    def __init__(self, name, int_lb, int_ub):
        """ Constructor. """
        self.name = name
        self.int_lb = int_lb
        self.int_ub = int_ub
        self.all_data = []
        self._model_initialised = False

    # Add/process data -----------------------------------------------------------------------------
    def update_model_with_new_data(self, Allocs, Rewards, Loads, Sigmas, Event_times):
        """ Updates model with the given data. """
        raise NotImplementedError('Implement in a child class.')

    def initialise_model(self):
        """ Initialise model. """
        if not self._model_initialised:
            self._initialise_model_child()
            self._model_initialised = True

    def _initialise_model_child(self):
        """ Initialise model. """
        raise NotImplementedError('Implement in a child class.')

    # Recommendations ------------------------------------------------------------------------------
    def get_recommendation(self, perf_goal, *args, **kwargs):
        """ Obtain recommendation. """
        raise NotImplementedError('Implement in a child class.')

    def get_recommendation_for_upper_bound(self, perf_goal, *args, **kwargs):
        """ Obtain recommendation. """
        raise NotImplementedError('Implement in a child class.')

    def get_recommendation_for_lower_bound(self, perf_goal, *args, **kwargs):
        """ Obtain recommendation. """
        raise NotImplementedError('Implement in a child class.')

    def compute_conf_interval_for_input(self, x):
        """ Compute a confidence interval for the input point x. """
        raise NotImplementedError('Implement in a child class.')

    def compute_estimate_for_input(self, x):
        """ Obtains an estimate (along with ucbs and lcbs) for the given point. """
        raise NotImplementedError('Implement in a child class.')


class BaseLearner(DemandRecommender):
    """ Base learner class. """

    def __init__(self, app_id, data_logger, model, concurrency_resolution_method='two_models'):
        """ Constructor. """
        self.app_id = app_id
        self._model_1 = model
        self.data_logger = data_logger
        self._training_thread_running = False
        self.concurrency_resolution_method = concurrency_resolution_method
        self.total_data_received = 0
#         self.update_on_copy_of_live_model = update_on_copy_of_live_model
        if self.concurrency_resolution_method == 'two_models':
            self._model_2 = deepcopy(model)
            self._model_1_being_updated = False
            self._model_2_being_updated = False
#             self._model_1_being_used = False
#             self._model_2_being_used = False
        elif concurrency_resolution_method == 'deepcopy':
            self._model_2 = deepcopy(model)
        else:
            self._model_2 = self._model_1
        self.current_time_stamp = None

    def get_model_for_serving(self):
        """ Returns the model for serving. """
        if self.concurrency_resolution_method == 'two_models':
            if not self._model_2_being_updated:
                return self._model_2
            else:
                return self._model_1
        elif self.concurrency_resolution_method == 'deepcopy':
            return self._model_2
        else:
            raise ValueError('Unknown concurrency_resolution_method %s.'%(
                self.concurrency_resolution_method))

    def initialise(self):
        """ Initialise. """
        self._model_1.initialise_model()
        if self.concurrency_resolution_method in ['two_models', 'deepcopy']:
            self._model_2.initialise_model()

    def _fetch_and_format_latest_data(self):
        """ Fetches and formats the latest data. """
        new_data, new_time_stamp = \
            self.data_logger.get_data(fields=['alloc', 'load', 'reward', 'sigma',
                                              'event_start_time', 'event_end_time'],
                                      start_time_stamp=self.current_time_stamp,
                                      end_time_stamp=None)
        Allocs = [elem['alloc'] for elem in new_data]
        Rewards = [elem['reward'] for elem in new_data]
        Loads = [elem['load'] for elem in new_data]
        Sigmas = [elem['sigma'] for elem in new_data]
        Event_times = [elem['event_end_time'] - elem['event_start_time'] for elem in new_data]
        return Allocs, Rewards, Loads, Sigmas, Event_times, new_time_stamp

    def model_update_routine(self):
        """ Run initialise when you are. """
        Allocs, Rewards, Loads, Sigmas, Event_times, new_time_stamp = \
            self._fetch_and_format_latest_data()
        num_data_received = len(Allocs)
        if num_data_received == 0:
            return
        logger.debug('Received %d data for model %s.' % (num_data_received, self._model_1.name))
        self.current_time_stamp = new_time_stamp # this order is important. we want to ignore the
                                                 # initial data when collecting.
        # Check if we should ignore the initial data points -------------------------------------
        self.total_data_received += num_data_received
        if (self.total_data_received - num_data_received) < NUM_INIT_DATA_TO_IGNORE:
            logger.info('%s returning since total data received is %d (%d)',
                        self.app_id, self.total_data_received - num_data_received,
                        self.total_data_received)
            return
        # Now add data to the model -------------------------------------------------------------
        if self.concurrency_resolution_method == 'two_models':
            # First update model 1
            self._model_1_being_updated = True
            self._model_1.update_model_with_new_data(Allocs, Rewards, Loads, Sigmas, Event_times)
            self._model_1_being_updated = False
            self._model_2_being_updated = True
            self._model_2.update_model_with_new_data(Allocs, Rewards, Loads, Sigmas, Event_times)
            self._model_2_being_updated = False
        elif self.concurrency_resolution_method == 'deepcopy':
            self._model_1.update_model_with_new_data(Allocs, Rewards, Loads, Sigmas, Event_times)
            model_deep_copy = deepcopy(self._model_1)
            self._model_2 = model_deep_copy
        else:
            raise ValueError('Unknown concurrency_resolution_method %s.'%(
                self.concurrency_resolution_method))

    # Recommendations ------------------------------------------------------------------------------
    def get_recommendation(self, perf_goal, load, *args, **kwargs):
        """ Obtain recommendation. """
        model_for_serving = self.get_model_for_serving()
        return load * model_for_serving.get_recommendation(perf_goal, *args, **kwargs)

    def get_recommendation_for_upper_bound(self, perf_goal, load, *args, **kwargs):
        """ Obtain recommendation. """
        model_for_serving = self.get_model_for_serving()
        return load * model_for_serving.get_recommendation_for_upper_bound(perf_goal,
                                                                           *args, **kwargs)

    def get_recommendation_for_lower_bound(self, perf_goal, load, *args, **kwargs):
        """ Obtain recommendation. """
        model_for_serving = self.get_model_for_serving()
        return load * model_for_serving.get_recommendation_for_lower_bound(perf_goal,
                                                                           *args, **kwargs)

    def compute_conf_interval_for_input(self, alloc, load):
        """ Compute a confidence interval for the input point x. """
        model_for_serving = self.get_model_for_serving()
        return model_for_serving.compute_conf_interval_for_input(alloc/load)

    def compute_estimate_for_input(self, alloc, load):
        """ Compute estimates (including lcb and ucb) for the allocation and load. """
        model_for_serving = self.get_model_for_serving()
        return model_for_serving.compute_estimate_for_input(alloc/load)

    def get_data_logger(self):
        """ Return the data logger. """
        return self.data_logger

    def plot_estimate(self, plot_data=None, to_show=True):
        """ Plots the estimate. """
        # pylint: disable=import-outside-toplevel
        import matplotlib.pyplot as plt
        import numpy as np
        model_for_serving = self.get_model_for_serving()
        if plot_data is None:
            plot_data = {}
        if 'test_inputs' in plot_data:
            test_inputs = plot_data['test_inputs']
        else:
            test_inputs = np.linspace(model_for_serving.int_lb, model_for_serving.int_ub, 200)
        data_inputs = [elem[0] for elem in model_for_serving.all_data]
        data_labels = [elem[1] for elem in model_for_serving.all_data]
        # Obtain confidence intervals
        conf_intervals = [model_for_serving.compute_conf_interval_for_input(x) for x in test_inputs]
        lcbs = [elem[0] for elem in conf_intervals]
        ucbs = [elem[1] for elem in conf_intervals]
        means = [0.5 * (elem[0] + elem[1]) for elem in conf_intervals]
        # Do the plotting --------------------------------------------------------------------------
        plt.figure(figsize=(26, 12))
        plt.fill_between(test_inputs, lcbs, ucbs, color='g', alpha=0.2)
        plt.plot(test_inputs, means, '--', color='b', linewidth=7)
        plt.scatter(data_inputs, data_labels, s=400, color='k', marker='x', linewidth=4)
        max_y = max(max(ucbs), max(data_labels))
        max_x = max(max(data_inputs), max(test_inputs))
        # Plot entitlement and demand
#         if 'entitlement' in plot_data:
#             entitlement_labels = [0, max_y]
#             entitlement_inputs = [plot_data['entitlement'], plot_data['entitlement']]
#             print('   - entitlement for %s: %0.4f'%(self.app_id, plot_data['entitlement']))
#             plt.plot(entitlement_inputs, entitlement_labels, linestyle='--', color='r',
#                      linewidth=4)
        if 'unit_demand' in plot_data:
            demand_labels = [0, plot_data['performance_goal']]
            demand_inputs = [plot_data['unit_demand'], plot_data['unit_demand']]
            demy_labels = [plot_data['performance_goal']] * 2
            demy_inputs = [0, plot_data['unit_demand']]
            print('   - unit demand (perf-goal) for %s: %0.4f (%0.4f)'%(
                self.app_id, plot_data['unit_demand'], plot_data['performance_goal']))
            plt.plot(demand_inputs, demand_labels, linestyle='--', color='magenta', linewidth=3)
            plt.plot(demy_inputs, demy_labels, linestyle='--', color='magenta', linewidth=3)
        plt.margins(0.01)
#         file_name = 'conf_int_%d'%(ag_idx) + '.pdf'
        plt.xticks(np.linspace(0, max_x, 5), fontsize=30)
        plt.yticks(np.linspace(0, max_y, 11), fontsize=30)
        if 'title' in plot_data:
            plt.title(plot_data['title'])
        if 'save_file_name' in plot_data:
            plt.savefig(plot_data['save_file_name'])
        if to_show:
            plt.show()


