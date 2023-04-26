"""
    A class for recording performances.
    -- romilbhardwaj
    -- kirthevasank
"""

import logging
import pickle
import threading
import time
import numpy as np
# Local
from cilantro.core.bank import Bank
from cilantro.core.fair_alloc_utils import egalitarian_welfare, fairness_violation, \
                                           resource_loss, utilitarian_welfare

logger = logging.getLogger(__name__)


class PerformanceRecorder:
    """ Performance recorder. """

    def __init__(self, app_id, performance_goal, util_scaling, unit_demand, entitlement,
                 data_logger):
        """ Constructor. """
        super().__init__()
        self.app_id = app_id
        self.performance_goal = performance_goal
        self.util_scaling = util_scaling
        self.unit_demand = unit_demand
        self.entitlement = entitlement
        self.data_logger = data_logger
        self._history = []
        self._most_recent_data = None
        # For computing and tracking the utility -------------------------------------------------
        self._vals_updated_with_recent_data = None
        self._curr_time_stamp = None
        self._curr_total_util = 0
        self._curr_total_sq_util = 0
        self._curr_total_reward = 0
        self._curr_total_sq_reward = 0
        self._curr_total_load = 0
        self._curr_total_sq_load = 0
        self._curr_total_alloc = 0
        self._curr_total_sq_alloc = 0
        self._curr_total_time = 0

    def set_entitlement(self, entitlement):
        """ Set entitlement. """
        self.entitlement = entitlement

    def compute_util_from_reward(self, reward):
        """ Computes the utility from the reward. """
        norm_val = min(1.0, reward/self.performance_goal)
        if self.util_scaling == 'linear':
            return norm_val
        elif self.util_scaling == 'quadratic':
            return norm_val ** 2
        elif self.util_scaling == 'sqrt':
            return np.sqrt(norm_val)
        else:
            return self.util_scaling(norm_val)

    def fetch_data_and_update_history(self):
        """ fetch data and update history. """
        new_data, new_time_stamp = self.data_logger.get_data(
            fields=['alloc', 'reward', 'load', 'event_end_time', 'event_start_time'],
            start_time_stamp=self._curr_time_stamp, end_time_stamp=None)
        logger.debug('Received %d new data in %s.', len(new_data), self.app_id)
        if new_data:
            self._history.extend(new_data)
            self._curr_time_stamp = new_time_stamp
            self._most_recent_data = new_data
            self._vals_updated_with_recent_data = False
        else:
            self._most_recent_data = []
        return new_data

    def get_most_recent_data(self):
        """ Returns the most recent data. """
        return self._most_recent_data

    def get_all_data(self):
        """ Returns the most recent data. """
        return self._history

    def compute_metrics_for_data_batch(self, data_batch):
        """ Computes the average utility and load for the new data. """
        total_reward = 0
        total_sq_reward = 0
        total_util = 0
        total_sq_util = 0
        total_load = 0
        total_sq_load = 0
        total_alloc = 0
        total_sq_alloc = 0
        total_alloc_per_unit_load = 0
        total_sq_alloc_per_unit_load = 0
        total_time = 0
        for dat in data_batch:
            curr_event_time = dat['event_end_time'] - dat['event_start_time']
            total_reward += dat['reward'] * curr_event_time
            total_sq_reward += (dat['reward'] ** 2) * curr_event_time
            inst_util = self.compute_util_from_reward(dat['reward'])
            total_util += inst_util * curr_event_time
            total_sq_util += (inst_util ** 2) * curr_event_time
            total_load += dat['load'] * curr_event_time
            total_sq_load += (dat['load'] ** 2) * curr_event_time
            total_alloc += dat['alloc'] * curr_event_time
            total_sq_alloc += (dat['alloc'] ** 2) * curr_event_time
            total_alloc_per_unit_load += (dat['alloc']/dat['load']) * curr_event_time
            total_sq_alloc += ((dat['alloc']/dat['load']) ** 2) * curr_event_time
            total_time += curr_event_time
        ret = {'total_reward': total_reward,
               'total_sq_reward': total_sq_reward,
               'total_util': total_util,
               'total_sq_util': total_sq_util,
               'total_load': total_load,
               'total_sq_load': total_sq_load,
               'total_alloc': total_alloc,
               'total_sq_alloc': total_sq_alloc,
               'total_alloc_per_unit_load': total_alloc_per_unit_load,
               'total_sq_alloc_per_unit_load': total_sq_alloc_per_unit_load,
               'total_time': total_time,
              }
        return ret

    @classmethod
    def _get_zero_metrics_dict(cls):
        """ Returns zero metric dict"""
        return {'mean_reward': 0, 'reward_std': 0,
                'mean_util': 0, 'util_std': 0,
                'mean_load': 0, 'load_std': 0,
                'mean_alloc': 0, 'alloc_std': 0,
                'total_time': 0}

    def _get_metrics_from_curr_status(self):
        """ Returns metrics from current status. """
        if self._curr_total_time <= 0.1:
            return self._get_zero_metrics_dict()
        mean_reward = self._curr_total_reward / self._curr_total_time
        mean_sq_reward = self._curr_total_sq_reward / self._curr_total_time
        reward_std = np.sqrt(mean_sq_reward - mean_reward ** 2)
        mean_util = self._curr_total_util / self._curr_total_time
        mean_sq_util = self._curr_total_sq_util / self._curr_total_time
        util_std = np.sqrt(mean_sq_util - mean_util ** 2)
        mean_load = self._curr_total_load / self._curr_total_time
        mean_sq_load = self._curr_total_sq_load / self._curr_total_time
        load_std = np.sqrt(mean_sq_load - mean_load ** 2)
        mean_alloc = self._curr_total_alloc / self._curr_total_time
        mean_sq_alloc = self._curr_total_sq_alloc / self._curr_total_time
        alloc_std = np.sqrt(mean_sq_alloc - mean_alloc ** 2)
        return {'mean_reward': mean_reward, 'reward_std': reward_std,
                'mean_util': mean_util, 'util_std': util_std,
                'mean_load': mean_load, 'load_std': load_std,
                'mean_alloc': mean_alloc, 'alloc_std': alloc_std,
                'total_time': self._curr_total_time}

    def _compute_updated_util_and_return_new_metrics(self, fetch_new_data=False):
        """ Computes the updated utility and returns the new metrics. """
        if fetch_new_data:
            new_data = self.fetch_data_and_update_history()
        else:
            new_data = self.get_most_recent_data()
        if new_data and (not self._vals_updated_with_recent_data):
            metrics_for_new_batch = self.compute_metrics_for_data_batch(new_data)
            self._curr_total_time += metrics_for_new_batch['total_time']
            self._curr_total_reward += metrics_for_new_batch['total_reward']
            self._curr_total_sq_reward += metrics_for_new_batch['total_sq_reward']
            self._curr_total_util += metrics_for_new_batch['total_util']
            self._curr_total_sq_util += metrics_for_new_batch['total_sq_util']
            self._curr_total_load += metrics_for_new_batch['total_load']
            self._curr_total_sq_load += metrics_for_new_batch['total_sq_load']
            self._curr_total_alloc += metrics_for_new_batch['total_alloc']
            self._curr_total_sq_alloc += metrics_for_new_batch['total_sq_alloc']
        else:
            metrics_for_new_batch = None
        return self._get_metrics_from_curr_status(), metrics_for_new_batch

    def compute_updated_util(self, fetch_new_data=False):
        """ Computes the utility. """
        ret, _ = self._compute_updated_util_and_return_new_metrics(fetch_new_data)
        return ret

    def compute_util(self, fetch_new_data=False):
        """ Computes the utility from beginning to now. """
        if fetch_new_data:
            self.fetch_data_and_update_history()
        metrics_for_all_data = self.compute_metrics_for_data_batch(self._history)
        self._curr_total_time = metrics_for_all_data['total_time']
        self._curr_total_reward = metrics_for_all_data['total_reward']
        self._curr_total_sq_reward = metrics_for_all_data['total_sq_reward']
        self._curr_total_util = metrics_for_all_data['total_util']
        self._curr_total_sq_util = metrics_for_all_data['total_sq_util']
        self._curr_total_load = metrics_for_all_data['total_load']
        self._curr_total_sq_load = metrics_for_all_data['total_sq_load']
        self._curr_total_alloc = metrics_for_all_data['total_alloc']
        self._curr_total_sq_alloc = metrics_for_all_data['total_sq_alloc']
        return self._get_metrics_from_curr_status()

    def compute_recent_util(self, num_recent_event_logs=-1, fetch_new_data=False):
        """ Computes the utility from beginning to now. """
        if fetch_new_data:
            self.compute_updated_util()
        if num_recent_event_logs > 0:
            metrics = self.compute_metrics_for_data_batch(self._history[-num_recent_event_logs:])
        else:
            _, metrics = self._compute_updated_util_and_return_new_metrics(fetch_new_data=False)
        if not metrics:
            return self._get_zero_metrics_dict()
        mean_reward = metrics['total_reward'] / metrics['total_time']
        mean_sq_reward = metrics['total_sq_reward'] / metrics['total_time']
        reward_std = np.sqrt(mean_sq_reward - mean_reward ** 2)
        mean_util = metrics['total_util'] / metrics['total_time']
        mean_sq_util = metrics['total_sq_util'] / metrics['total_time']
        util_std = np.sqrt(mean_sq_util - mean_util ** 2)
        mean_load = metrics['total_load'] / metrics['total_time']
        mean_sq_load = metrics['total_sq_load'] / metrics['total_time']
        load_std = np.sqrt(mean_sq_load - mean_load ** 2)
        mean_alloc = metrics['total_alloc'] / metrics['total_time']
        mean_sq_alloc = metrics['total_sq_alloc'] / metrics['total_time']
        alloc_std = np.sqrt(mean_sq_alloc - mean_alloc ** 2)
        return {'mean_reward': mean_reward, 'reward_std': reward_std,
                'mean_util': mean_util, 'util_std': util_std,
                'mean_load': mean_load, 'load_std': load_std,
                'mean_alloc': mean_alloc, 'alloc_std': alloc_std,
                'total_time': self._curr_total_time}


class PerformanceRecorderBank(Bank):
    """ A bank for recording performances of different jobs. """

    def __init__(self, resource_quantity, alloc_granularity=None,
                 report_results_every=-1, save_file_name=None, report_results_descr=''):
        """ Constructor.
            - report_results_every is the time (in seconds) interval at which the results must be
              reported (and saved).
            - resource_quantity, alloc_granularity are the total number of resources and the
              allocation granularity.
            - save_file_name, if a string, is where the results need to be saved.
        """
        super().__init__()
        self.report_results_every = report_results_every
        self.resource_quantity = resource_quantity
        self.alloc_granularity = alloc_granularity
        self.save_file_name = save_file_name
        self.history = []
        self._curr_total_time = 0
        self._curr_avg_resource_loss = 0
        self._curr_avg_mean_fairness_viol = 0
        self._curr_avg_max_fairness_viol = 0
        self._curr_total_resource_loss = 0
        self._curr_total_sq_resource_loss = 0
        self._curr_total_mean_fairness_viol = 0
        self._curr_total_sq_mean_fairness_viol = 0
        self._curr_total_max_fairness_viol = 0
        self._curr_total_sq_max_fairness_viol = 0
        if self.report_results_every > 0:
            self._report_results_poll_time = self.report_results_every / 10
            self._last_report_time = None
            self._reporting_loop_running = False
            self._report_results_descr = report_results_descr

    @classmethod
    def _check_type(cls, obj):
        """ Checks type. """
        assert isinstance(obj, PerformanceRecorder)

    def initiate_report_results_loop(self):
        """ Initiates the report results loop. """
        self._last_report_time = time.time()
        if not self._reporting_loop_running:
            self._reporting_loop_running = True
            thread = threading.Thread(target=self._report_results_loop, args=())
            thread.start()

    def stop_report_results_loop(self):
        """ Stops the report results loop. """
        self._reporting_loop_running = False

    def _report_results_loop(self):
        """ This loop reports results in a separate thread. """
        # pylint: disable=broad-except
        while self._reporting_loop_running:
            time.sleep(self._report_results_poll_time)
            curr_time = time.time()
            if curr_time >= self._last_report_time + self.report_results_every:
                try:
                    results = self.compute_recent_results()
#                     print_str = (
#                         ('%s (%0.1f):: res-loss=%0.3f+-%0.3f, fair_viol:(sum=%0.3f, max=%0.3f, ' +
#                          'mean=%0.3f), util_welfare=%0.3f+-%0.3f, egal_welfare=%0.3f+-%0.3f')%(
#                              self._report_results_descr, results['time_elapsed'],
#                              results['resource_loss'][0], results['resource_loss'][1],
#                              results['sum_fairness_viol'][0],
#                              results['max_fairness_viol'][0], results['mean_fairness_viol'][0],
#                              results['util_welfare'][0], results['util_welfare'][1],
#                              results['egal_welfare'][0], results['egal_welfare'][1],
#                             ))
                    print_str = (
                        ('%s (%0.1f):: res-loss=%0.3f, fair_viol:(sum=%0.3f, max=%0.3f, ' +
                         'mean=%0.3f), util_welf=%0.3f, egal_welf=%0.3f, ' +
                         'avg_rew=%0.3f,  cost=%0.3f')%(
                             self._report_results_descr, results['time_elapsed'],
                             results['resource_loss'][0],
                             results['sum_fairness_viol'][0],
                             results['max_fairness_viol'][0], results['mean_fairness_viol'][0],
                             results['util_welfare'][0], results['egal_welfare'][0],
                             results['avg_reward'][0], results['cost'][0])
                            )
                    print_str = '\n' + print_str + '\n    - ' + results['leaf_utils_descr']
                    logger.info(print_str)
                except Exception as e:
                    logger.info('%s(ts=%0.2f): Recent results could not be computed. Exception: %s',
                                self._report_results_descr, curr_time, e)
                self._last_report_time = curr_time

    # Some utilities we will find useful --------------------------------------------------------
    def _update_history_of_all_leafs(self):
        """ updates history. """
        for _, pr in self.enumerate():
            pr.fetch_data_and_update_history()

    def _add_to_history_and_save_results(self, results, info=''):
        """ Saves results. """
        save_time = time.time()
        results_to_append = (save_time, info, results)
        self.history.append(results_to_append)
        if not (self.save_file_name is None):
            with open(self.save_file_name, 'wb') as pickle_save_file:
                pickle.dump(self.history, pickle_save_file)
                pickle_save_file.close()

    def _fetch_and_prepare_util_results(self, results_fetcher):
        """ Computes the updated average utility. """
        all_utils = []
        all_rewards = []
        all_allocs = []
        all_loads = []
        leaf_utils_descr_strs = []
        for tag, pr in self.enumerate():
            results = results_fetcher(pr)
            all_rewards.append(results['mean_reward'])
            all_utils.append(results['mean_util'])
            all_allocs.append(results['mean_alloc'])
            all_loads.append(results['mean_load'])
            curr_app_str = \
                '%s: util=%0.3f, rew=%0.3f alloc=%0.3f, load=%0.3f, ud=%0.3f, time=%0.3f'%(
                    tag, results['mean_util'], results['mean_reward'], results['mean_alloc'],
                    results['mean_load'], self.get(tag).unit_demand, results['total_time'])
            leaf_utils_descr_strs.append(curr_app_str)
        leaf_utils_descr = ', '.join(leaf_utils_descr_strs)
        results = {'avg_util': (np.mean(all_utils), np.std(all_utils)),
                   'avg_reward': (np.mean(all_rewards), np.std(all_rewards)),
                   'avg_load': (np.mean(all_loads), np.std(all_loads)),
                   'avg_alloc': (np.mean(all_allocs), np.std(all_allocs)),
                   'leaf_utils_descr': leaf_utils_descr}
        return results

    def compute_updated_results(self):
        """ Computes the updated average utility. """
        # 1. Update the history -------------------------------------------------------------
        self._update_history_of_all_leafs()
        # 2. Compute the losses -------------------------------------------------------------
        ret = self._compute_updated_losses()
        # 3. Compute the utilities ----------------------------------------------------------
        def _get_results_fetcher():
            """ Fetches results from performance recorder. """
            return lambda pr: pr.compute_updated_util(fetch_new_data=False)
        util_results = self._fetch_and_prepare_util_results(_get_results_fetcher())
        for key, val in util_results.items():
            ret[key] = val
        self._add_to_history_and_save_results(ret, 'updated_results')
        return ret

    def compute_results(self):
        """ Computes the updated average utility. """
        # 1. Update the history -------------------------------------------------------------
        self._update_history_of_all_leafs()
        # 2. Compute the losses -------------------------------------------------------------
        ret = self._compute_losses()
        # 3. Compute the utilities ----------------------------------------------------------
        def _get_results_fetcher():
            """ Fetches results from performance recorder. """
            return lambda pr: pr.compute_util(fetch_new_data=False)
        util_results = self._fetch_and_prepare_util_results(_get_results_fetcher())
        for key, val in util_results.items():
            ret[key] = val
        self._add_to_history_and_save_results(ret, 'from_scratch_results')
        return ret

    def compute_recent_results(self, num_recent_event_logs=-1, fetch_new_data=False):
        """ Computes the updated average utility. """
        # 1. Update the history -------------------------------------------------------------
        self._update_history_of_all_leafs()
        # 2. Compute the losses -------------------------------------------------------------
        ret = self._compute_recent_losses(num_recent_event_logs)
        # 3. Compute the utilities ----------------------------------------------------------
        def _get_results_fetcher(_num_recent_event_logs, _fetch_new_data):
            """ Fetches results from performance recorder. """
            return lambda pr: pr.compute_recent_util(_num_recent_event_logs, _fetch_new_data)
        util_results = self._fetch_and_prepare_util_results(
            _get_results_fetcher(num_recent_event_logs, fetch_new_data))
        for key, val in util_results.items():
            ret[key] = val
        self._add_to_history_and_save_results(ret, 'recent_results_%d'%(num_recent_event_logs))
        return ret

    # Utilities for computing the fairness violation and resource loss ---------------------------
    def _compute_updated_losses(self, **kwargs):
        """ Computes the losses for the data batch. """
        data_batches = {}
        for tag, pr in self.enumerate():
            data_batches[tag] = pr.get_most_recent_data()
        new_losses = self.compute_losses_on_batch_of_data(data_batches, **kwargs)
        if new_losses:
            total_time = self._curr_total_time + new_losses['time_period']
            self._curr_avg_resource_loss = \
                (self._curr_avg_resource_loss * self._curr_total_time +
                 new_losses['time_period'] * new_losses['resource_loss']) / total_time
            self._curr_avg_mean_fairness_viol = \
                (self._curr_avg_mean_fairness_viol * self._curr_total_time +
                 new_losses['time_period'] * new_losses['mean_fairness_viol']) / total_time
            self._curr_avg_max_fairness_viol = \
                (self._curr_avg_max_fairness_viol * self._curr_total_time +
                 new_losses['time_period'] * new_losses['max_fairness_viol']) / total_time
            self._curr_total_time = total_time
        ret = {'resource_loss': self._curr_avg_resource_loss,
               'mean_fairness_viol': self._curr_avg_mean_fairness_viol,
               'max_fairness_viol': self._curr_avg_max_fairness_viol,
               'time_period': self._curr_total_time,
               'time_elapsed': self._curr_total_time}
        # TODO: fix this to return the standard deviations and welfares.
        return ret

    def _compute_losses(self, **kwargs):
        """ Computes the losses for the data batch. """
        data_batches = {}
        for tag, pr in self.enumerate():
            data_batches[tag] = pr.get_all_data()
        losses = self.compute_losses_on_batch_of_data(data_batches, **kwargs)
        if losses:
            self._curr_avg_resource_loss = losses['resource_loss']
            self._curr_avg_mean_fairness_viol = losses['mean_fairness_viol']
            self._curr_avg_max_fairness_viol = losses['max_fairness_viol']
            self._curr_total_time = losses['time_period']
        ret = {'resource_loss': self._curr_avg_resource_loss,
               'mean_fairness_viol': self._curr_avg_mean_fairness_viol,
               'max_fairness_viol': self._curr_avg_max_fairness_viol,
               'time_period': self._curr_total_time,
               'time_elapsed': self._curr_total_time}
        # TODO: fix this to return the standard errors as welfares.
        return ret

    def _compute_recent_losses(self, num_recent_event_logs, **kwargs):
        """ Computes the losses for the data batch. """
        data_batches = {}
        for tag, pr in self.enumerate():
            if num_recent_event_logs > 0:
                data_batches[tag] = pr.get_all_data()[-num_recent_event_logs:]
            else:
                data_batches[tag] = pr.get_most_recent_data()
        losses = self.compute_losses_on_batch_of_data(data_batches, **kwargs)
        if losses:
            self._curr_total_time += losses['time_period']
            ret = {'resource_loss': losses['resource_loss'],
                   'sum_fairness_viol': losses['sum_fairness_viol'],
                   'mean_fairness_viol': losses['mean_fairness_viol'],
                   'max_fairness_viol': losses['max_fairness_viol'],
                   'util_welfare': losses['util_welfare'],
                   'egal_welfare': losses['egal_welfare'],
                   'avg_reward': losses['avg_reward'],
                   'cost': losses['cost'],
                   'time_period': losses['time_period'],
                   'time_elapsed': self._curr_total_time,
                  }
        else:
            ret = {'resource_loss': (np.nan, np.nan),
                   'mean_fairness_viol': (np.nan, np.nan),
                   'max_fairness_viol': (np.nan, np.nan),
                   'util_welfare': (np.nan, np.nan),
                   'egal_welfare': (np.nan, np.nan),
                   'time_period': (np.nan, np.nan),
                   'time_elapsed': np.nan,
                  }
        return ret

    def get_recent_allocs_and_util_metrics(self, num_recent_event_logs, grid_size):
        """ Computes recent allocations and util metrics. """
        data_batches = {}
        for tag, pr in self.enumerate():
            if num_recent_event_logs > 0:
                data_batches[tag] = pr.get_all_data()[-num_recent_event_logs:]
        losses = self.compute_losses_on_batch_of_data(data_batches, grid_size=grid_size,
                                                      return_grid_vals=True)
        if not losses['grid_vals']:
            return None
        else:
            ret = []
            for idx, alloc in enumerate(losses['grid_vals']['all_allocs']):
                curr_val = {
                    'alloc': dict(zip(losses['grid_vals']['leaf_order'], alloc)),
                    'loads': dict(zip(losses['grid_vals']['leaf_order'],
                                      losses['grid_vals']['all_loads'][idx])),
                    'leaf_utils': dict(zip(losses['grid_vals']['leaf_order'],
                                           losses['grid_vals']['all_utils'][idx])),
                    'leaf_rewards': dict(zip(losses['grid_vals']['leaf_order'],
                                             losses['grid_vals']['all_rewards'][idx])),
                    'util_welfare': losses['grid_vals']['util_welfare'][idx],
                    'egal_welfare': losses['grid_vals']['egal_welfare'][idx],
                    }
                ret.append(curr_val)
        return ret

    @classmethod
    def _compute_mean_values_in_intervals_in_sorted_batch(cls, sorted_batches, grid, fields):
        """ Computes the mean allocation and demand. """
        # The following function helps us split the data into different time intervals -----------
        def _get_threshold_idx(leaf, start_idx, grid_interval_end_time):
            """ Returns the index which captures grid_interval_end_time. """
            ret_idx = start_idx
            num_data = len(sorted_batches[leaf])
            while sorted_batches[leaf][ret_idx]['event_end_time'] < grid_interval_end_time and \
                  ret_idx < num_data:
                ret_idx += 1
            return ret_idx
        # This helps compute the mean values in different time intervals --------------------------
        def _compute_mean_value_in_set_of_intervals(start_time, end_time, curr_batch_vals):
            """ Computes the mean value in the set of intervals. """
            tot_time = 0
            tot_vals = {fld: 0 for fld in fields}
            len_curr_batch = len(curr_batch_vals)
            # Go through each element in curr_batch_vals and compute the value for each field -----
            for idx, bv in enumerate(curr_batch_vals):
                if idx == 0:
                    curr_time = max(0, (min(curr_batch_vals[0]['event_end_time'], end_time) -
                                        max(curr_batch_vals[0]['event_start_time'], start_time))
                                   )
                elif idx == len_curr_batch - 1:
                    curr_time = max(0, (min(curr_batch_vals[-1]['event_end_time'], end_time) -
                                        max(curr_batch_vals[-1]['event_start_time'], start_time))
                                   )
                else:
                    curr_time = bv['event_end_time'] - bv['event_start_time']
                tot_time += curr_time
                for fld in fields:
                    tot_vals[fld] += curr_time * bv[fld]
            # Finally compute the average ---------------------------------------------------------
            ret = {fld: val/tot_time for fld, val in tot_vals.items()}
            return ret, tot_time
        # Start computation here ------------------------------------------------------------------
        grid_size = len(grid) - 1
        ret = {leaf: {fld: [] for fld in fields} for leaf in sorted_batches}
        for leaf, leaf_batch in sorted_batches.items():
            curr_grid_idx = 0
#             all_grid_idxs = [curr_grid_idx]
            prev_leaf_vals = {fld: 1 for fld in fields} # TODO: do something more intelligent here!
            for idx in range(grid_size):
                next_grid_idx = _get_threshold_idx(leaf, curr_grid_idx, grid[idx+1])
                curr_leaf_batch = leaf_batch[curr_grid_idx: next_grid_idx + 1]
                # Compute values for the leaf in this batch ------------------------------------
                curr_leaf_vals, curr_time = _compute_mean_value_in_set_of_intervals(
                    grid[idx], grid[idx+1], curr_leaf_batch)
                if curr_time == 0:
                    curr_leaf_vals = prev_leaf_vals
                for fld, mean_val in curr_leaf_vals.items():
                    ret[leaf][fld].append(mean_val)
                curr_grid_idx = next_grid_idx
                prev_leaf_vals = curr_leaf_vals
#                 all_grid_idxs.append(curr_grid_idx)
        return ret

    def compute_losses_on_batch_of_data(self, data_batches, grid_size=100, return_grid_vals=False):
        """ Computes losses on batches of data. data_batches is a dictionary mapping the tag
            to a list of data points.
        """
        # pylint: disable=too-many-statements
        # pylint: disable=too-many-locals
        leaf_data_lengths = {key: len(val) for key, val in data_batches.items()}
        if len(data_batches) == 0 or min(leaf_data_lengths.values()) == 0:
            logger.info('Returning zeros since len(data_batches)=%d, leaf_data_lengths: %s',
                        len(data_batches), leaf_data_lengths)
            ret = {'resource_loss': (0, 0),
                   'mean_fairness_viol': (0, 0),
                   'max_fairness_viol': (0, 0),
                   'sum_fairness_viol': (0, 0),
                   'util_welfare': (0, 0),
                   'egal_welfare': (0, 0),
                   'avg_reward': (0, 0),
                   'cost': (0, 0),
                   'time_period': 0}
            if return_grid_vals:
                ret['grid_vals'] = None
            return ret
        # Preliminaries ---------------------------------------------------------------------------
        all_leafs = self.get_tags()
        unit_demands_available = [(self.get(leaf).unit_demand is not None) for leaf in all_leafs]
        if (not all_leafs) or (not all(unit_demands_available)):
            return None
        # First sort all batches of data according to the start time ------------------------------
        sorted_batches = {}
        for tag in data_batches:
            sorted_batches[tag] = \
                sorted(data_batches[tag], key=lambda elem: elem['event_start_time'])
            if not np.isfinite(sorted_batches[tag][-1]['event_end_time']):
                sorted_batches[tag] = sorted_batches[tag][:-1]
        # Compute the start and finish times ------------------------------------------------------
        start_time = np.max([val[0]['event_start_time'] for _, val in sorted_batches.items()])
        end_time = np.min([val[-1]['event_end_time'] for _, val in sorted_batches.items()])
        grid = np.linspace(start_time, end_time, grid_size+1)
        vals_for_loss_comp = self._compute_mean_values_in_intervals_in_sorted_batch(
            sorted_batches, grid, fields=['alloc', 'load', 'reward'])
        vals_for_loss_comp_by_grid = [{} for _ in range(grid_size)]
        for idx, elem in enumerate(vals_for_loss_comp_by_grid):
            elem['allocs'] = [vals_for_loss_comp[leaf]['alloc'][idx] for leaf in all_leafs]
            elem['loads'] = [vals_for_loss_comp[leaf]['load'][idx] for leaf in all_leafs]
            elem['rewards'] = [vals_for_loss_comp[leaf]['reward'][idx] for leaf in all_leafs]
            elem['utils'] = [
                self.get(leaf).compute_util_from_reward(vals_for_loss_comp[leaf]['reward'][idx])
                for leaf in all_leafs]
            elem['demands'] = [vals_for_loss_comp[leaf]['load'][idx] * self.get(leaf).unit_demand
                               for leaf in all_leafs]
        leaf_entitlements = [self.get(leaf).entitlement * self.resource_quantity
                             for leaf in all_leafs]
        # Now compute the losses and average ------------------------------------------------------
        resource_losses = [resource_loss(elem['demands'], elem['allocs'], self.resource_quantity)
                           for elem in vals_for_loss_comp_by_grid]
        util_welfares = [utilitarian_welfare(elem['utils']) for elem in vals_for_loss_comp_by_grid]
        egal_welfares = [egalitarian_welfare(elem['utils']) for elem in vals_for_loss_comp_by_grid]
        all_allocs = [elem['allocs'] for elem in vals_for_loss_comp_by_grid]
        sum_costs = [sum(elem) for elem in all_allocs]
        all_rewards = [elem['rewards'] for elem in vals_for_loss_comp_by_grid]
        avg_rewards = [np.mean(elem) for elem in all_rewards]
        all_utils = [elem['utils'] for elem in vals_for_loss_comp_by_grid]
        all_loads = [elem['loads'] for elem in vals_for_loss_comp_by_grid]
#         all_demands = [elem['demands'] for elem in vals_for_loss_comp_by_grid]
#         print('all_allocs', all_allocs)
#         print('sum_allocs', [sum(elem) for elem in all_allocs])
#         import pdb; pdb.set_trace()
        mean_fairness_viols = []
        max_fairness_viols = []
        sum_fairness_viols = []
        for elem in vals_for_loss_comp_by_grid:
            curr_sum_fv, curr_mean_fv, curr_max_fv = fairness_violation(
                elem['demands'], elem['allocs'], leaf_entitlements,
                self.resource_quantity, alloc_granularity=self.alloc_granularity)
            mean_fairness_viols.append(curr_mean_fv)
            max_fairness_viols.append(curr_max_fv)
            sum_fairness_viols.append(curr_sum_fv)
        ret = {'resource_loss': (np.mean(resource_losses), np.std(resource_losses)),
               'mean_fairness_viol': (np.mean(mean_fairness_viols), np.std(mean_fairness_viols)),
               'max_fairness_viol': (np.mean(max_fairness_viols), np.std(max_fairness_viols)),
               'sum_fairness_viol': (np.mean(sum_fairness_viols), np.std(sum_fairness_viols)),
               'util_welfare': (np.mean(util_welfares), np.std(util_welfares)),
               'egal_welfare': (np.mean(egal_welfares), np.std(egal_welfares)),
               'avg_reward': (np.mean(avg_rewards), np.std(avg_rewards)),
               'cost': (np.mean(sum_costs), np.std(sum_costs)),
               'time_period': end_time - start_time}
        if return_grid_vals:
            ret['grid_vals'] = {}
            ret['grid_vals']['grid'] = grid[1:]
            ret['grid_vals']['resource_loss'] = resource_losses
            ret['grid_vals']['mean_fairness_viol'] = mean_fairness_viols
            ret['grid_vals']['max_fairness_viol'] = max_fairness_viols
            ret['grid_vals']['sum_fairness_viol'] = sum_fairness_viols
            ret['grid_vals']['util_welfare'] = util_welfares
            ret['grid_vals']['egal_welfare'] = egal_welfares
            ret['grid_vals']['all_allocs'] = all_allocs
            ret['grid_vals']['all_utils'] = all_utils
            ret['grid_vals']['all_loads'] = all_loads
            ret['grid_vals']['all_rewards'] = all_rewards
            ret['grid_vals']['sum_costs'] = sum_costs
            ret['grid_vals']['leaf_order'] = all_leafs
        return ret

