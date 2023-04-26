"""
    Some utilities for plotting time series info.

    -- kirthevasank
"""

import os
import numpy as np
import matplotlib.pyplot as plt
# Local imports
from cilantro.ancillary.plotting_duplicate import read_experiment_info_from_files, \
                                                  process_log_files_in_single_workdir
from cilantro.profiling.profiled_info_loader import ProfiledInfoBank



NUM_INITIAL_ENTRIES_TO_IGNORE = 10

TIME_ELAPSED = 'time_elapsed'
RESOURCE_LOSS = 'resource_loss'
MAX_FAIRNESS_VIOL = 'max_fairness_viol'
MEAN_FAIRNESS_VIOL = 'mean_fairness_viol'
SUM_FAIRNESS_VIOL = 'sum_fairness_viol'
UTIL_WELFARE = 'util_welfare'
EGAL_WELFARE = 'egal_welfare'
AVG_UTIL = 'avg_util'
LEAF_REWARDS = 'all_rewards'
LEAF_COSTS = 'all_allocs'

ALLOC_PER_UNIT_LOAD = 'alloc_per_unit_load'
REWARD = 'reward'
EVENT_TIME_PERIOD = 'event_time_period'
EVENT_START_TIME = 'event_start_time'
EVENT_END_TIME = 'event_end_time'
SIGMA = 'sigma'
LOAD = 'load'
ALLOC = 'alloc'

BAR_FIGSIZE = (26, 12)
BAR_FIGSIZE_WITH_LEGEND = (26, 12)
BAR_LEGEND_FONT_SIZE = 30
BAR_ANNOT_FONT_SIZE = 25
BAR_AXIS_FONT_SIZE = 30
BAR_YLABEL_FONT_SIZE = 26
CURVE_FIG_SIZE = (20, 12)


def get_plot_options(options, **kwargs):
    """ Obtain plot options. """
    if options is None:
        options = {}
    def _check_for_value_and_set(key, default_value):
        """ Checks for value. """
        if key in options:
            return options[key]
        elif key in kwargs:
            return kwargs[key]
        else:
            return default_value
    ret = {}
    ret['num_grid_pts'] = _check_for_value_and_set('num_grid_pts', 100)
    ret['err_bar_type'] = _check_for_value_and_set('err_bar_type', 'fill_between')
    ret['line_width'] = _check_for_value_and_set('line_width', 4)
    ret['legend_location'] = _check_for_value_and_set('legend_location', 'upper left')
    ret['legend_font_size'] = _check_for_value_and_set('legend_font_size', 30)
    ret['to_plot_bar_legend'] = _check_for_value_and_set('to_plot_bar_legend', True)
    ret['title_font_size'] = _check_for_value_and_set('title_font_size', 30)
    ret['axis_font_size'] = _check_for_value_and_set('axis_font_size', 70)
    ret['xtick_font_size'] = _check_for_value_and_set('xtick_font_size', 35)
    ret['ytick_font_size'] = _check_for_value_and_set('ytick_font_size', 35)
    ret['marker_size'] = _check_for_value_and_set('marker_size', 40 * (4**3))
    ret['util_compute_grid_size'] = _check_for_value_and_set('util_compute_grid_size', 1000)
    ret['fld_to_title'] = _check_for_value_and_set('fld_to_title',
        {RESOURCE_LOSS: 'Effective resource wastage',
         MAX_FAIRNESS_VIOL: 'NJC Fairness (max)',
         MEAN_FAIRNESS_VIOL: 'NJC Fairness (mean)', AVG_UTIL: 'Average Utility',
         SUM_FAIRNESS_VIOL: 'NJC Fairness (sum)',
         UTIL_WELFARE: 'Sociale welfare',
         EGAL_WELFARE: 'Egalitarian welfare',
         LEAF_REWARDS: 'Reward',
         LEAF_COSTS: 'Total Cost'})
    ret['plot_save_dir'] = _check_for_value_and_set('plot_save_dir', None)
    for fld in options:
        if not (fld in ret):
            ret[fld] = options[fld]
    return ret


def _get_workdirs_from_parent_directory_and_env_name(parent_dir, env_descr):
    """ Return a list of working directories. """
    dir_list = [elem for elem in os.listdir(parent_dir) if env_descr in elem]
    ret = [os.path.join(parent_dir, elem) for elem in dir_list]
    return ret


def load_results_file(policy_work_dir):
    """ Loads results file. """
    env, experiment_info = read_experiment_info_from_files(policy_work_dir)
    all_files_in_dir = os.listdir(policy_work_dir)
    pickle_files_in_dir = [elem for elem in all_files_in_dir if elem.endswith('.p')]
    in_run_file = pickle_files_in_dir[0]
    user_log_files = [os.path.join(policy_work_dir, elem) for elem in all_files_in_dir
                      if elem.endswith('.csv')]
#     print('pickle_files_in_dir', pickle_files_in_dir)
#     print('user_log_files', user_log_files)
    return env, experiment_info, user_log_files, in_run_file


def gen_ts_curve_from_metrics(env_metrics, policy_name, save_fig_dir,
                              field_legend_marker_dict, options):
    """ Generates time series curve from the metrics. """
    # pylint: disable=import-outside-toplevel
    # pylint: disable=multiple-statements
    # pylint: disable=unused-variable
    # pylint: disable=unused-argument
    print(options)
    grid = env_metrics['grid_vals']['grid']
    time_grid = np.linspace(options['time_min'], options['time_max'], options['time_grid_size'])
    field_vals = {}
    mean_val = {}
    fig, ax = plt.subplots(figsize=(20, 14))
    for fld in field_legend_marker_dict:
        field_vals[fld] = np.interp(time_grid, env_metrics['grid_vals']['grid'],
                                    env_metrics['grid_vals'][fld])
        plt.plot(time_grid, field_vals[fld], color=field_legend_marker_dict[fld]['colour'],
                 linestyle=field_legend_marker_dict[fld]['linestyle'],
                 label=field_legend_marker_dict[fld]['legend'],
                 linewidth=options['line_width'])
        mean_val[fld] = np.mean(field_vals[fld])
        fig.tight_layout()
    fig_name = 'ts_plots/%s.pdf'%(policy_name)
    fig.savefig(fig_name)
    plt.xlabel('Time')
#     plt.title(policy_name)
    plt.legend(loc='lower right', fontsize=40)
    print('Method: %s', policy_name)
    for fld, val in field_legend_marker_dict.items():
        print('  %s: %0.4f', fld, val)


def plot_alloc_timeseries(work_dir_name, plot_from, policy_name,
                          profiled_info_dir, save_fig_dir, field_legend_marker_dict,
                          options=None, **kwargs):
#                  method_legend_colour_marker_dict, x_bounds=None, to_plot_legend=True,
#                  plot_type='plot', options=None, **kwargs):
    """ Plot allocation time series. """
    options = get_plot_options(options, **kwargs)
    all_policy_work_dirs = \
        _get_workdirs_from_parent_directory_and_env_name(work_dir_name[0], work_dir_name[1])
    curr_policy_workdir = [elem for elem in all_policy_work_dirs if policy_name in elem][0]
#     print(curr_policy_workdir)
    env, experiment_info, user_log_files, _ = load_results_file(curr_policy_workdir)
    if plot_from == 'logs':
        profiled_info_bank = ProfiledInfoBank(profiled_info_dir)
        env_metrics, _, _ = \
            process_log_files_in_single_workdir(
                env, user_log_files, profiled_info_bank, experiment_info,
                util_compute_grid_size=options['util_compute_grid_size'])
        gen_ts_curve_from_metrics(env_metrics, policy_name, save_fig_dir,
                                  field_legend_marker_dict, options)
        plt.draw()
        plt.show()
    else:
        raise ValueError('Unknown argumenet for plot_from: %s.'%(plot_from))


