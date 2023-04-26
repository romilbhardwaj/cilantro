"""
    Some utilities for plotting.
    Adapted from Dragonfly
    (github.com/dragonfly/dragonfly/blob/master/dragonfly/utils/plot_utils.py)
    -- kirthevasank
"""

# pylint: disable=too-many-arguments
# pylint: disable=too-many-statements
# pylint: disable=too-many-locals
# pylint: disable=too-many-branches

import os
import pickle
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
# Local
from cilantro.ancillary.info_write_load_utils import read_experiment_info_from_files
from cilantro.core.henv import are_two_environments_equal
from cilantro.core.performance_recorder import PerformanceRecorder, PerformanceRecorderBank
from cilantro.profiling.profiled_info_loader import ProfiledInfoBank


# matplotlib.rcParams['mathtext.fontset'] = 'custom'
# matplotlib.rcParams['mathtext.rm'] = 'Bitstream Vera Sans'
# matplotlib.rcParams['mathtext.it'] = 'Bitstream Vera Sans:italic'
# matplotlib.rcParams['mathtext.bf'] = 'Bitstream Vera Sans:bold'
# matplotlib.rcParams['mathtext.fontset'] = 'stix'
# matplotlib.rcParams['font.family'] = 'STIXGeneral'

NUM_INITIAL_ENTRIES_TO_IGNORE = 15

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

BAR_FIGSIZE = (26, 4.2)
BAR_FIGSIZE_WITH_LEGEND = (26, 6.2)
BAR_XTICK_FONTSIZE = 25
BAR_YTICK_FONTSIZE = 15
BAR_LEGEND_FONT_SIZE = 30
BAR_ANNOT_FONT_SIZE = 25
BAR_AXIS_FONT_SIZE = 30
BAR_YLABEL_FONT_SIZE = 26
CURVE_FIG_SIZE = (20, 12)

ENV_CURVE_KEYS = [RESOURCE_LOSS, SUM_FAIRNESS_VIOL, MEAN_FAIRNESS_VIOL, MAX_FAIRNESS_VIOL,
                  UTIL_WELFARE, EGAL_WELFARE]
AUTOSCALING_CURVE_KEYS = [LEAF_REWARDS, LEAF_COSTS]
# ENV_CURVE_KEYS = [RESOURCE_LOSS, SUM_FAIRNESS_VIOL, UTIL_WELFARE, EGAL_WELFARE]
# ENV_CURVE_KEYS = [RESOURCE_LOSS, MAX_FAIRNESS_VIOL, UTIL_WELFARE, EGAL_WELFARE]
# TABLE_METRICS_TO_PRINT = [RESOURCE_LOSS, MAX_FAIRNESS_VIOL, UTIL_WELFARE, EGAL_WELFARE]
TABLE_METRICS_TO_PRINT = ENV_CURVE_KEYS


ALL_FIELDS_INRUN = [RESOURCE_LOSS, MAX_FAIRNESS_VIOL, MEAN_FAIRNESS_VIOL, AVG_UTIL]
ALL_FIELDS_INRUN_AND_TIME = ALL_FIELDS_INRUN + [TIME_ELAPSED]


def rgba(red, green, blue, a):
    """rgba: generates matplotlib compatible rgba values from html-style rgba values
    """
    return (red / 255.0, green / 255.0, blue / 255.0, a)


def transparent(red, green, blue, _, opacity=0.5):
    """ Transparent: converts a rgba color to a transparent opacity.
    """
    return (red, green, blue, opacity)


def to_hex(hexstring):
    """to_hex: generates matplotlib-compatible rgba values from html-style hex colours.
    """
    if hexstring[0] == '#':
        hexstring = hexstring[1:]
    red = int(hexstring[:2], 16)
    green = int(hexstring[2:4], 16)
    blue = int(hexstring[4:], 16)
    return rgba(red, green, blue, 1.0)


COLOURS = {
    'aqua': to_hex('#7FDBFF'),
    'black': to_hex('#111111'),
    'blue': to_hex('#0074D9'),
    'brown': to_hex('#964B00'),
    'cyan': to_hex('#00FFFF'),
    'darkgreen': to_hex('#006400'),
    'fuchsia': to_hex('#F012BE'),
    'green': to_hex('#2ECC40'),
    'grey': to_hex('#AAAAAA'),
    'red': to_hex('#FF4136'),
    'teal': to_hex('#39CCCC'),
    'lightgreen': to_hex('#ADFF2F'),
    'lightsalmon': to_hex('#FFA07A'),
    'lime': to_hex('#01FF70'),
    'magenta': to_hex('#FF00FF'),
    'maroon': to_hex('#85144B'),
    'navy': to_hex('#001F3F'),
    'olive': to_hex('#3D9970'),
    'orange': to_hex('#FF851B'),
    'purple': to_hex('#B10DC9'),
    'silver': to_hex('#DDDDDD'),
    'white': to_hex('#FFFFFF'),
    'yellow': to_hex('#FFDC00'),
}


# Utilities for loading results and plotting -----------------------------------------------------
def get_data_from_data_log_file(file_name):
    """ Process file. """
    df = pd.read_csv(file_name, index_col=0)
    ret = []
    row_counter = 0
    for _, row in df.iterrows():
        row_counter += 1
        if row_counter >= NUM_INITIAL_ENTRIES_TO_IGNORE:
            curr_data = {EVENT_TIME_PERIOD: row['event_end_time'] - row['event_start_time'],
                         REWARD: row['reward'],
                         ALLOC_PER_UNIT_LOAD: row['alloc']/row['load'],
                         ALLOC: row['alloc'],
                         SIGMA: row['sigma'],
                         EVENT_START_TIME: row['event_start_time'],
                         EVENT_END_TIME: row['event_end_time'],
                         LOAD: row['load'],
                        }
            ret.append(curr_data)
    ret.sort(key= lambda elem: elem['event_start_time'])
#     print('Retrieved %d (ignored %d) rows from %s'%(
#         row_counter, NUM_INITIAL_ENTRIES_TO_IGNORE, file_name))
    return ret
#     earliest_event_start_time = ret[0]['event_start_time']
#     for elem in ret:
#         elem[REL_EVENT_START_TIME] = elem[EVENT_START_TIME] - earliest_event_start_time
#     return ret, earliest_event_start_time


def compute_leaf_metrics_from_individual_experiment_metrics(
        list_of_leaf_metrics_per_experiment):
    """ Returns metrics from current status. """
    fields = ['total_reward', 'total_sq_reward',
              'total_util', 'total_sq_util',
              'total_load', 'total_sq_load',
              'total_alloc', 'total_sq_alloc',
              'total_alloc_per_unit_load', 'total_sq_alloc_per_unit_load',
              'total_time',
             ]
    ret = {}
    for leaf in list_of_leaf_metrics_per_experiment[0]:
        leaf_metrics = {}
        for fld in fields:
            leaf_metrics[fld] = sum(elem[leaf][fld] for elem in list_of_leaf_metrics_per_experiment)
        mean_reward = leaf_metrics['total_reward'] / leaf_metrics['total_time']
        mean_sq_reward = leaf_metrics['total_sq_reward'] / leaf_metrics['total_time']
        mean_reward_std = np.sqrt(mean_sq_reward - mean_reward ** 2)
        leaf_metrics['mean_reward'] = (mean_reward, mean_reward_std)
        mean_util = leaf_metrics['total_util'] / leaf_metrics['total_time']
        mean_sq_util = leaf_metrics['total_sq_util'] / leaf_metrics['total_time']
        mean_util_std = np.sqrt(mean_sq_util - mean_util ** 2)
        leaf_metrics['mean_util'] = (mean_util, mean_util_std)
        mean_load = leaf_metrics['total_load'] / leaf_metrics['total_time']
        mean_sq_load = leaf_metrics['total_sq_load'] / leaf_metrics['total_time']
        mean_load_std = np.sqrt(mean_sq_load - mean_load ** 2)
        leaf_metrics['mean_load'] = (mean_load, mean_load_std)
        mean_alloc = leaf_metrics['total_alloc'] / leaf_metrics['total_time']
        mean_sq_alloc = leaf_metrics['total_sq_alloc'] / leaf_metrics['total_time']
        mean_alloc_std = np.sqrt(mean_sq_alloc - mean_alloc ** 2)
        leaf_metrics['mean_alloc'] = (mean_alloc, mean_alloc_std)
        mean_alloc_per_unit_load = \
            leaf_metrics['total_alloc_per_unit_load'] / leaf_metrics['total_time']
        mean_sq_alloc_per_unit_load = \
            leaf_metrics['total_sq_alloc_per_unit_load'] / leaf_metrics['total_time']
        mean_alloc_per_unit_load_std = \
            np.sqrt(mean_sq_alloc_per_unit_load - mean_alloc_per_unit_load ** 2)
        leaf_metrics['mean_alloc_per_unit_load'] = \
            (mean_alloc_per_unit_load, mean_alloc_per_unit_load_std)
        ret[leaf] = leaf_metrics
    return ret


def process_log_files_in_single_workdir(env, log_files_in_workdir, profiled_info_bank,
                                        experiment_info, util_compute_grid_size=1000):
    """ Processes the log files in the single workdir. """
    performance_recorder_bank = PerformanceRecorderBank(
        resource_quantity=experiment_info['resource_quantity'],
        alloc_granularity=experiment_info['alloc_granularity'])
    assert len(log_files_in_workdir) == len(env.leaf_nodes)
    entitlements = env.get_entitlements()
    experiment_data = {}
    experiment_leaf_metrics = {}
    for log_file_name in log_files_in_workdir:
        lfn_wo_suffix = log_file_name.split('/')[-1][:-4] # remove csv
        components = lfn_wo_suffix.split('_')
        leaf_path = components[0]
        workload_type = components[1]
        leaf = env.leaf_nodes[leaf_path]
        leaf_unit_demand = profiled_info_bank.get(workload_type).get_alloc_for_payoff_vals(
            [leaf.threshold])[0]
        performance_recorder = PerformanceRecorder(
            app_id=leaf_path, performance_goal=leaf.threshold, unit_demand=leaf_unit_demand,
            util_scaling=leaf.util_scaling, entitlement=entitlements[leaf_path], data_logger=None)
        performance_recorder_bank.register(leaf_path, performance_recorder)
        experiment_data[leaf_path] = get_data_from_data_log_file(log_file_name)
        experiment_leaf_metrics[leaf_path] = performance_recorder.compute_metrics_for_data_batch(
            experiment_data[leaf_path])
    # Compute environment metrics ------------------------------------------------------------------
    env_metrics_table = performance_recorder_bank.compute_losses_on_batch_of_data(
        experiment_data, grid_size=util_compute_grid_size, return_grid_vals=True)
    env_metrics = env_metrics_table['grid_vals']
    print('env_metrics', type(env_metrics))
    env_metrics['grid'] = [elem - env_metrics['grid'][0] for elem in env_metrics['grid']]
#     import pdb;pdb.set_trace()
    # Correct fairness -----------------------------------------------------------------------
    fairness_metric_keys = [MEAN_FAIRNESS_VIOL, MAX_FAIRNESS_VIOL, SUM_FAIRNESS_VIOL]
    for fmk in fairness_metric_keys:
        env_metrics_table[fmk] = (1 - env_metrics_table[fmk][0], env_metrics_table[fmk][1])
        env_metrics[fmk] = [(1 - elem) for elem in env_metrics[fmk]]
    # Return -----------------------------------------------------------------------
    return env_metrics_table, env_metrics, experiment_leaf_metrics


def autolabel(rects, ax, font_col):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = np.round(rect.get_height(), 3)
        ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height * 0.5),
                    xy=(rect.get_x() + rect.get_width() / 2, 0.05),
                    xytext=(0, 0),  # 3 points vertical offset
                    textcoords="offset points", color=font_col,
                    ha='center', va='bottom', fontsize=BAR_ANNOT_FONT_SIZE,
                    rotation=90)


def create_bar_plot(metric_name, metric_results, leafs_to_plot, plot_order,
                    method_legend_colour_marker_dict, plot_legend=True, plot_save_dir=None):
    """ Create bar plot. """
    if not plot_order:
        plot_order = metric_results.keys()
    num_leafs = len(leafs_to_plot)
    num_methods = len(plot_order)
    bar_width = 0.79 / num_methods
    # Create bar plot of utils -------------------------------------------------------------------
    if plot_legend:
        fig, ax = plt.subplots(figsize=BAR_FIGSIZE_WITH_LEGEND)
    else:
        fig, ax = plt.subplots(figsize=BAR_FIGSIZE)
    X = np.arange(num_leafs)
    all_rects = [None] * num_methods
    max_y_lim = 1
    for bar_idx, method in enumerate(plot_order):
        bar_means = [metric_results[method][leaf][0] for leaf in leafs_to_plot]
        bar_std_errs = [metric_results[method][leaf][1] for leaf in leafs_to_plot]
        bar_label = method_legend_colour_marker_dict[method]['legend']
        bar_col = method_legend_colour_marker_dict[method]['colour']
        max_y_lim = max(max_y_lim, max(bar_means))
        all_rects[bar_idx] = ax.bar(X + bar_idx * bar_width - (num_methods - 1) * bar_width/2,
                                    bar_means, bar_width, label=bar_label, color=bar_col,
                                    yerr=[bar_std_errs, bar_std_errs])
    y_lim = max(1.1, min(max_y_lim, 1.5))
    ax.set_ylim(0, y_lim)
    ax.set_ylabel(metric_name, fontsize=BAR_YLABEL_FONT_SIZE)
    leaf_labels = [''] + leafs_to_plot
    ax.set_xticks(np.arange(-1, len(leaf_labels)-1, step=1))
    plt.yticks(fontsize=BAR_YTICK_FONTSIZE)
    ax.set_xticklabels(leaf_labels, fontsize=BAR_XTICK_FONTSIZE)
#     ax.set_xlim([-0.5, X[-1] + 0.5])
    if plot_legend:
        ax.legend(loc="lower center", bbox_to_anchor=(0, 1.02, 1, 0.2), ncol=num_methods//2,
                  prop={'size': BAR_LEGEND_FONT_SIZE})
    for idx, rects in enumerate(all_rects):
        bar_label = method_legend_colour_marker_dict[plot_order[idx]]['legend']
        bar_col = method_legend_colour_marker_dict[plot_order[idx]]['colour']
        if bar_col in ['b', 'k']:
            font_col = 'white'
        else:
            font_col = 'k'
        autolabel(rects, ax, font_col)
    fig.tight_layout()
    if plot_save_dir:
        save_file_name = '%s/bar_%s'%(plot_save_dir, metric_name)
        plt.savefig(save_file_name + '.pdf')
        plt.savefig(save_file_name + '.png')


def create_bar_plots_from_leaf_metrics(
    leaf_metrics_for_each_method, plot_order, method_legend_colour_marker_dict,
    leafs_to_plot, options):
    """ Creates bar plots from leaf metrics. """
    metrics_to_plot = ['mean_util', 'mean_alloc', 'mean_alloc_per_unit_load']
    for metric_name in metrics_to_plot:
        metric_results = {}
        for method, method_results in leaf_metrics_for_each_method.items():
            metric_results[method] = {leaf:method_results[leaf][metric_name]
                                      for leaf in leafs_to_plot}
        create_bar_plot(metric_name, metric_results, leafs_to_plot, plot_order,
                        method_legend_colour_marker_dict, plot_legend=options['to_plot_bar_legend'],
                        plot_save_dir=options['plot_save_dir'])


def process_log_files_lists_for_a_method(env, work_dir_log_files_for_method, profiled_info_bank,
                                         experiment_info, util_compute_grid_size=1000):
    """ Processes log files for a method. """
    all_env_metrics = []
    all_env_table_metrics = []
    all_leaf_metrics = []
    for log_files_in_workdir in work_dir_log_files_for_method:
        curr_env_metrics_final, curr_env_metrics, curr_experiment_leaf_metrics = \
            process_log_files_in_single_workdir(env, log_files_in_workdir, profiled_info_bank,
                                                experiment_info, util_compute_grid_size)
        all_env_metrics.append(curr_env_metrics)
        all_leaf_metrics.append(curr_experiment_leaf_metrics)
        all_env_table_metrics.append(curr_env_metrics_final)
    # Process the leaf metrics --------------------------------------------------------------------
    processed_leaf_metrics = compute_leaf_metrics_from_individual_experiment_metrics(
        all_leaf_metrics)
    return all_env_metrics, all_env_table_metrics, processed_leaf_metrics


def compute_final_table_env_metrics_from_individual_metrics(all_env_table_metrics,
                                                            method_order, to_include_stderr=False):
    """ Final environment table metrics from individual metrics. """
    pd_columns = []
    for metric in TABLE_METRICS_TO_PRINT:
        if to_include_stderr:
            pd_columns.extend([metric + ':mean', metric + ':stderr'])
        else:
            pd_columns.extend([metric + ':mean'])
    df = pd.DataFrame(columns=pd_columns)
    for meth in method_order:
        meth_metrics = all_env_table_metrics[meth]
        curr_row = []
        for metric in TABLE_METRICS_TO_PRINT:
            curr_vals = [elem[metric][0] for elem in meth_metrics]
            curr_stdsqs = [elem[metric][1] ** 2 for elem in meth_metrics]
            curr_total_time = sum([elem['time_period'] for elem in meth_metrics])
            curr_mean = np.mean(curr_vals)
            curr_stderr = np.sqrt(np.sum(curr_stdsqs) / curr_total_time)
            if to_include_stderr:
                curr_row.extend([curr_mean, curr_stderr])
            else:
                curr_row.extend([curr_mean])
        df.loc[meth] = curr_row
    return df


def process_log_files_lists_for_all_methods(env, data_log_files, profiled_info_bank,
                                            experiment_info, method_order,
                                            util_compute_grid_size=1000):
    """ Processes log files for all methods. """
    env_metrics = {}
    all_env_table_metrics = {}
    leaf_metrics = {}
    for meth, work_dir_log_files_for_method in data_log_files.items():
        print('method', meth)
        meth_env_metrics, meth_env_table_metrics, meth_leaf_metrics = \
            process_log_files_lists_for_a_method(env, work_dir_log_files_for_method,
                                                 profiled_info_bank, experiment_info,
                                                 util_compute_grid_size)
        env_metrics[meth] = meth_env_metrics
        leaf_metrics[meth] = meth_leaf_metrics
        all_env_table_metrics[meth] = meth_env_table_metrics
    processed_final_env_table_metrics = compute_final_table_env_metrics_from_individual_metrics(
        all_env_table_metrics, method_order)
    return env_metrics, processed_final_env_table_metrics, leaf_metrics


def read_results_from_in_run_file(file_path):
    """ Reads results from a pickle file.
        file_path: the path to the file.
        returns: a dataframe object with all the various pieces of data.
    """
    try:
        with open(file_path, 'rb') as pickle_file:
            history = pickle.load(pickle_file)
            pickle_file.close()
        history.sort(key = lambda x: x[2][TIME_ELAPSED])
        results = {}
        for field in ALL_FIELDS_INRUN_AND_TIME:
            results[field] = [elem[2][field] for elem in history]
        results[TIME_ELAPSED] = [val - results[TIME_ELAPSED][0] for val in results[TIME_ELAPSED]]
        return results
    except:
        return None


def get_plot_info_from_inrun_results(results_from_files, grid_pts):
    """ Generates means and standard deviation for the method's output.
    """
    num_grid_pts = len(grid_pts)
    num_experiments = len(results_from_files)
    grid_vals_mean = {fld: np.zeros((num_experiments, num_grid_pts)) for fld in ALL_FIELDS_INRUN}
    grid_vals_std = {fld: np.zeros((num_experiments, num_grid_pts)) for fld in ALL_FIELDS_INRUN}
    # Iterate through each experiment -----------------------------------------------------------
    grid_within_bounds = True
    for exp_iter, exp_result in enumerate(results_from_files):
        if exp_result[TIME_ELAPSED][-1] < grid_pts[-1]:
            grid_within_bounds = False
        for fld in ALL_FIELDS_INRUN:
            exp_mean_vals = [elem[0] for elem in exp_result[fld]]
            interp_mean = np.interp(grid_pts, exp_result[TIME_ELAPSED], exp_mean_vals)
            grid_vals_mean[fld][exp_iter, :] = interp_mean
            exp_std_vals = [elem[1] for elem in exp_result[fld]]
            interp_std = np.interp(grid_pts, exp_result[TIME_ELAPSED], exp_std_vals)
            grid_vals_std[fld][exp_iter, :] = interp_std
    if not grid_within_bounds:
        print('Grid not within bounds!')
    # An internal function to get mean and std ------------------------------------------------
    ret_means = {fld: grid_vals_mean[fld].mean(axis=0) for fld in ALL_FIELDS_INRUN}
    mean_std2 = {fld: grid_vals_mean[fld].std(axis=0) ** 2 for fld in ALL_FIELDS_INRUN}
    grid_std2 = {fld: (grid_vals_std[fld] ** 2).mean(axis=0) for fld in ALL_FIELDS_INRUN}
    ret_std = {fld: np.sqrt(mean_std2[fld] + grid_std2[fld])/np.sqrt(2) for fld in ALL_FIELDS_INRUN}
    ret = {}
    for fld in ALL_FIELDS_INRUN:
        ret[fld] = (ret_means[fld], ret_std[fld])
    return ret


# def _get_grid_from_x_bounds(x_bounds, options):
#     """ returns a grid from x_bounds and options. """
#     if x_bounds is None:
#         min_last_time_val = np.inf
#         for meth in plot_order:
#             min_last_time_val = min(
#                 min_last_time_val,
#                 min(elem[TIME_ELAPSED][-1] for elem in in_run_results_from_files[meth]))
#         x_bounds = (0, min_last_time_val)
#     grid_pts = np.linspace(x_bounds[0], x_bounds[1], num=options['num_grid_pts'])
#     return grid_pts


def _get_grid_pt_vals_for_experiment_curve(exp_curve, grid_pts):
    """ Get grid points for experiment curve. """
    ret = {}
    for curve_type in ENV_CURVE_KEYS:
        ret[curve_type] = np.interp(grid_pts, exp_curve['grid'], exp_curve[curve_type])
    return ret

def _get_grid_pt_vals_for_method(experiment_curves_for_method, grid_pts):
    """ Returns the grid points for the method. """
    grid_vals_for_experiments = {curve_type:[] for curve_type in ENV_CURVE_KEYS}
    for ecfm in experiment_curves_for_method:
        curr_grid_vals = _get_grid_pt_vals_for_experiment_curve(ecfm, grid_pts)
        for curve_type in ENV_CURVE_KEYS:
            grid_vals_for_experiments[curve_type].append(curr_grid_vals[curve_type])
    ret = {}
    for curve_type, all_expt_grid_vals in grid_vals_for_experiments.items():
        curves_as_grid = np.array(all_expt_grid_vals)
        mean_est = np.mean(curves_as_grid, axis=0)
        if curves_as_grid.shape[0] > 1:
            std_err = np.std(curves_as_grid, axis=0) / np.sqrt(curves_as_grid.shape[0])
        else:
            std_err = np.zeros((len(grid_pts), ))
#         mean_est = list(mean_est)
#         std_err = list(std_err)
        ret[curve_type] = (mean_est, std_err)
    return ret


def _get_grid_pt_vals_for_each_method(env_metrics, grid_pts):
    """ Obtain grid point vals for each method. """
    grid_vals_for_each_method = {}
    for meth, experiment_curves_for_method in env_metrics.items():
        grid_vals_for_each_method[meth] = _get_grid_pt_vals_for_method(experiment_curves_for_method,
                                                                       grid_pts)
    return grid_vals_for_each_method


def _get_grid_pt_vals_for_autoscaling(env_metrics, grid_pts, fields_to_plot, plot_order):
    """ Get grid point values for auto scaling. """
    leaf_paths = list(env_metrics[plot_order[0]][0]['leaf_order'])
    ret = {lp: {} for lp in leaf_paths}
    for leaf_idx, leaf in enumerate(leaf_paths):
        leaf_dict = {fld: {} for fld in env_metrics}
        for fld in fields_to_plot:
            fld_dict = {}
            for meth in plot_order:
                curr_meth_data = []
                num_experiments = len(env_metrics[meth])
                for exp_idx in range(num_experiments):
                    # First collect the current data ---------------------------------------
                    curr_exp_data = [elem[leaf_idx] for elem in env_metrics[meth][exp_idx][fld]]
                    # Compute the grid -----------------------------------------------------
                    curr_exp_grid = env_metrics[meth][exp_idx]['grid']
                    curr_exp_grid_pts = np.interp(grid_pts, curr_exp_grid, curr_exp_data)
                    curr_meth_data.append(curr_exp_grid_pts)
                # Store the mean and std ----------------------------------------------------
                curr_meth_data = np.array(curr_meth_data)
                curr_meth_means = curr_meth_data.mean(axis=0)
                if curr_meth_data.shape[0] > 1:
                    curr_meth_stderrs = curr_meth_data.std(axis=0) / \
                                        np.sqrt(curr_meth_data.shape[0])
                else:
                    curr_meth_stderrs = np.zeros(curr_meth_means.shape)
                fld_dict[meth] = (curr_meth_means, curr_meth_stderrs)
            leaf_dict[fld] = fld_dict
        ret[leaf] = leaf_dict
    return ret


def gen_auto_scaling_curves_from_env_metrics(env_metrics,
                                             plot_order,
                                             fields_to_plot,
                                             method_legend_colour_marker_dict,
                                             options,
                                             x_bounds,
                                             plot_type,
                                             to_plot_legend):
    """ Generate auto scaling curves. """
    # pylint: disable=unused-argument
    # pylint: disable=unused-variable
    if x_bounds is None:
        min_last_time_val = np.inf
        for meth in plot_order:
            min_last_time_val = min(
                min_last_time_val,
                min(elem['grid'][-1] for elem in env_metrics[meth]))
        x_bounds = (0, min_last_time_val)
    grid_pts = np.linspace(x_bounds[0], x_bounds[1], num=options['num_grid_pts'])
    as_grid_vals = _get_grid_pt_vals_for_autoscaling(env_metrics, grid_pts,
                                                     fields_to_plot, plot_order)
    # Set plot type  -----------------------------------------------------------------------------
    if plot_type == 'plot':
        plot_func = plt.plot
    elif plot_type == 'loglog':
        plot_func = plt.loglog
    elif plot_type == 'semilogy':
        plot_func = plt.semilogy
    elif plot_type == 'semilogx':
        plot_func = plt.semilogx
    else:
        raise ValueError('Unknown plot function.')
    plt.rc('xtick', labelsize=options['xtick_font_size'])
    plt.rc('ytick', labelsize=options['ytick_font_size'])
    # Go through each leaf_path in the environment -----------------------------------------------
    ret_metrics = {}
    for leaf_path in as_grid_vals:
        leaf_grid_vals = as_grid_vals[leaf_path]
        for fld in fields_to_plot: # go through each plot field ------------------------------------
            plt.figure(figsize=CURVE_FIG_SIZE)
            for method in plot_order:
                if options['err_bar_type'] == 'fill_between':
                    plt.fill_between(
                        grid_pts, leaf_grid_vals[fld][method][0] - leaf_grid_vals[fld][method][1],
                        leaf_grid_vals[fld][method][0] + leaf_grid_vals[fld][method][1],
                        color=transparent(
                            *COLOURS[method_legend_colour_marker_dict[method]['colour']],
                            opacity=0.3),
                    )
                else:
                    raise NotImplementedError('Not implemented yet!')
            # Now plot the whole curve
            for method in plot_order:
                plot_func(grid_pts, leaf_grid_vals[fld][method][0],
                          marker=',',
                          color=method_legend_colour_marker_dict[method]['colour'],
                          linestyle=method_legend_colour_marker_dict[method]['linestyle'],
                          linewidth=options['line_width'],
                          label=method_legend_colour_marker_dict[method]['legend'],
                          )
            # Wrapping up -------------------------------------------------------------------------
            if to_plot_legend:
                plt.legend(loc=options['legend_location'], fontsize=options['legend_font_size'])
            plt.title(options['fld_to_title'][fld], fontsize=options['title_font_size'])
            plt.xlabel('Time', fontsize=options['axis_font_size'])
            if fld == LEAF_COSTS:
                plt.ylim((0, 60))
            # Save file --------------------------------------------------------
            if options['plot_save_dir']:
                save_file_name = os.path.join(options['plot_save_dir'], fld + '.png')
                plt.savefig(save_file_name)
        # Return the final metrics in a pandas data frame ------------------------------------------
        pd_columns = []
#         to_include_stderr = True
        to_include_stderr = False
        for metric in fields_to_plot:
            if to_include_stderr:
                pd_columns.extend([metric + ':mean', metric + ':stderr'])
            else:
                pd_columns.extend([metric + ':mean'])
        df = pd.DataFrame(columns=pd_columns)
        for meth in plot_order:
            curr_row = []
            for metric in fields_to_plot:
    #             import pdb; pdb.set_trace()
                curr_vals = leaf_grid_vals[metric][meth][0]
                len_vals = len(curr_vals)
                curr_mean = np.mean(curr_vals)
                curr_stderr = np.std(curr_vals) / np.sqrt(len_vals)
                if to_include_stderr:
                    curr_row.extend([curr_mean, curr_stderr])
                else:
                    curr_row.extend([curr_mean])
            df.loc[meth] = curr_row
        ret_metrics[leaf_path] = df
    return ret_metrics


def gen_curves_from_env_metrics(env_metrics,
                                plot_order,
                                fields_to_plot,
                                method_legend_colour_marker_dict,
                                options,
                                x_bounds=None,
                                plot_type='semilogy',
                                to_plot_legend=True,
                                ):
    """ Generate curves from environment metrics. """
    if x_bounds is None:
        min_last_time_val = np.inf
        for meth in plot_order:
            min_last_time_val = min(
                min_last_time_val,
                min(elem['grid'][-1] for elem in env_metrics[meth]))
        x_bounds = (0, min_last_time_val)
    grid_pts = np.linspace(x_bounds[0], x_bounds[1], num=options['num_grid_pts'])
    grid_vals_for_each_method = _get_grid_pt_vals_for_each_method(env_metrics, grid_pts)
    return _gen_curve_from_processed_results_dict(
        grid_vals_for_each_method, grid_pts, fields_to_plot, plot_order,
        method_legend_colour_marker_dict, options, plot_type, to_plot_legend)


def gen_curves_from_in_run_results(in_run_results_from_files,
                                   plot_order,
                                   method_legend_colour_marker_dict,
                                   options,
                                   x_bounds=None,
                                   plot_type='semilogy',
                                   to_plot_legend=True,
                                   ):
    """ Plots the curves given the experiment results.
    """
    # Compute grid -------------------------------------------------------------------------------
    if x_bounds is None:
        min_last_time_val = np.inf
        for meth in plot_order:
            min_last_time_val = min(
                min_last_time_val,
                min(elem[TIME_ELAPSED][-1] for elem in in_run_results_from_files[meth]))
        x_bounds = (0, min_last_time_val)
    grid_pts = np.linspace(x_bounds[0], x_bounds[1], num=options['num_grid_pts'])
    # Compute means and stds ---------------------------------------------------------------------
    results_dict = {}
    for meth in plot_order:
        results_dict[meth] = get_plot_info_from_inrun_results(in_run_results_from_files[meth],
                                                              grid_pts)
    return _gen_curve_from_processed_results_dict(
        results_dict, grid_pts, ALL_FIELDS_INRUN, plot_order, method_legend_colour_marker_dict,
        options, plot_type, to_plot_legend)


def _gen_curve_from_processed_results_dict(results_dict, grid_pts, plot_fields, plot_order,
        method_legend_colour_marker_dict, options, plot_type, to_plot_legend):
    """ Generates curve from processed results dictionary. """
    # Set plot type  -----------------------------------------------------------------------------
    if plot_type == 'plot':
        plot_func = plt.plot
    elif plot_type == 'loglog':
        plot_func = plt.loglog
    elif plot_type == 'semilogy':
        plot_func = plt.semilogy
    elif plot_type == 'semilogx':
        plot_func = plt.semilogx
    else:
        raise ValueError('Unknown plot function.')
    plt.rc('xtick', labelsize=options['xtick_font_size'])
    plt.rc('ytick', labelsize=options['ytick_font_size'])
    # Plot each field --------------------------------------------------------------------
    for fld in plot_fields:
        plt.figure(figsize=CURVE_FIG_SIZE)
        for method in plot_order:
            if options['err_bar_type'] == 'fill_between':
                plt.fill_between(
                    grid_pts, results_dict[method][fld][0] - results_dict[method][fld][1],
                    results_dict[method][fld][0] + results_dict[method][fld][1],
                    color=transparent(*COLOURS[method_legend_colour_marker_dict[method]['colour']],
                                      opacity=0.3),
                )
            else:
                raise NotImplementedError('Not implemented yet!')
        # Now plot the whole curve
        for method in plot_order:
            plot_func(grid_pts, results_dict[method][fld][0],
                      marker=',',
                      color=method_legend_colour_marker_dict[method]['colour'],
                      linestyle=method_legend_colour_marker_dict[method]['linestyle'],
                      linewidth=options['line_width'],
                      label=method_legend_colour_marker_dict[method]['legend'],
                      )
        # Wrapping up -----------------------------------------------------------------------------
        if to_plot_legend:
            plt.legend(loc=options['legend_location'], fontsize=options['legend_font_size'])
        plt.title(options['fld_to_title'][fld], fontsize=options['title_font_size'])
        plt.xlabel('Time', fontsize=options['axis_font_size'])
        # Save file --------------------------------------------------------
        if options['plot_save_dir']:
            save_file_name = os.path.join(options['plot_save_dir'], fld + '.png')
            plt.savefig(save_file_name)
    # Return the final metrics in a pandas data frame ------------------------------------------
    pd_columns = []
    to_include_stderr = False
    for metric in ENV_CURVE_KEYS:
        if to_include_stderr:
            pd_columns.extend([metric + ':mean', metric + ':stderr'])
        else:
            pd_columns.extend([metric + ':mean'])
    df = pd.DataFrame(columns=pd_columns)
    for meth in plot_order:
        meth_metrics = results_dict[meth]
        curr_row = []
        for metric in ENV_CURVE_KEYS:
#             import pdb; pdb.set_trace()
            curr_vals = meth_metrics[metric][0]
            len_vals = len(curr_vals)
            curr_mean = np.mean(curr_vals)
            curr_stderr = np.std(curr_vals) / np.sqrt(len_vals)
            if to_include_stderr:
                curr_row.extend([curr_mean, curr_stderr])
            else:
                curr_row.extend([curr_mean])
        df.loc[meth] = curr_row
    # Create scatter plot for welfare vs fairness ------------------------------------------
    fairness_metrics_for_scatter = ['sum_fairness_viol:mean',
                                    'max_fairness_viol:mean',
                                    'mean_fairness_viol:mean']
    print("options['marker_size']", options['marker_size'])
    num_methods = len(plot_order)
    for fmfs in fairness_metrics_for_scatter:
        _, (ax_util, ax_egal) = plt.subplots(1, 2, figsize=CURVE_FIG_SIZE)
        for meth in plot_order:
            marker = '^' if meth in ['mmf', 'egalwelforacle', 'utilwelforacle'] else 'o'
            ax_util.scatter([df.loc[meth, fmfs]], [df.loc[meth, 'util_welfare:mean']],
                            color=method_legend_colour_marker_dict[meth]['colour'],
                            s=[options['marker_size']],
                            label=method_legend_colour_marker_dict[meth]['legend'])
            ax_egal.scatter([df.loc[meth, fmfs]], [df.loc[meth, 'egal_welfare:mean']],
                            color=method_legend_colour_marker_dict[meth]['colour'],
                            s=[options['marker_size']])
        ax_util.set_xlabel(fmfs, fontsize=options['axis_font_size'])
        ax_util.set_ylabel(options['fld_to_title'][UTIL_WELFARE],
                           fontsize=options['axis_font_size'])
        ax_egal.set_xlabel(fmfs, fontsize=options['axis_font_size'])
        ax_egal.set_ylabel(options['fld_to_title'][EGAL_WELFARE],
                           fontsize=options['axis_font_size'])
        ax_util.legend(loc="lower center", bbox_to_anchor=(0.25, 1.02, 0.8, 0.2),
                       ncol=num_methods//2,
                       prop={'size': 15})
    return df


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
    ret['axis_font_size'] = _check_for_value_and_set('axis_font_size', 30)
    ret['xtick_font_size'] = _check_for_value_and_set('xtick_font_size', 20)
    ret['ytick_font_size'] = _check_for_value_and_set('ytick_font_size', 20)
    ret['marker_size'] = _check_for_value_and_set('marker_size', 20 * (4**3))
    ret['util_compute_grid_size'] = _check_for_value_and_set('util_compute_grid_size', 1000)
    ret['fld_to_title'] = _check_for_value_and_set('fld_to_title',
        {RESOURCE_LOSS: 'Effective resource wastage', MAX_FAIRNESS_VIOL: 'NJC Fairness (max)',
         MEAN_FAIRNESS_VIOL: 'NJC Fairness (mean)', AVG_UTIL: 'Average Utility',
         SUM_FAIRNESS_VIOL: 'NJC Fairness (sum)',
         UTIL_WELFARE: 'Utilitarian Welfare (sum of utils)',
         EGAL_WELFARE: 'Egalitarian Welfare (max-min utility)',
         LEAF_REWARDS: 'Reward',
         LEAF_COSTS: 'Total Cost'})
    ret['plot_save_dir'] = _check_for_value_and_set('plot_save_dir', None)
    return ret


def load_results_files_from_workdirs(work_dirs):
    """ Loads all results files. """
    ret_data_logs = {}
    ret_results_from_files = {}
    first_env_as_list = []
    first_info_as_list = []
    def _process_files_in_curr_work_dir(dir_name, meth):
        """ Processes the current file. """
        # Preliminaries, create key if not present and load environment -------------------------
        all_files_in_dir = os.listdir(dir_name)
        if not (meth in ret_results_from_files):
            ret_results_from_files[meth] = []
        if not (meth in ret_data_logs):
            ret_data_logs[meth] = []
        curr_env, curr_info = read_experiment_info_from_files(dir_name)
        if first_env_as_list:
            first_env = first_env_as_list[0]
            if not are_two_environments_equal(first_env[1], curr_env):
                report_str = 'Environments not equal:\n%s:\n%s\n\n%s:\n%s'%(
                    first_env[0], first_env[1].write_to_file(None),
                    dir_name, curr_env.write_to_file(None))
                raise ValueError(report_str)
            first_info = first_info_as_list[0]
            if not curr_info == first_info:
                report_str = 'Experimental info are not equal:\n%s: %s\n%s: %s'%(
                    first_env[0], first_info, dir_name, curr_info)
                raise ValueError(report_str)
        else:
            first_env_as_list.append((dir_name, curr_env))
            first_info_as_list.append(curr_info)
        # Next load the in-run results -------------------------------------------------------------
        pickle_files_in_dir = [elem for elem in all_files_in_dir if elem.endswith('.p')]
        if pickle_files_in_dir:
            in_run_results_file = os.path.join(dir_name, pickle_files_in_dir[0])
            ret_results_from_files[meth].append(read_results_from_in_run_file(in_run_results_file))
        else:
            ret_results_from_files[meth].append(None)
        # Next load the data logs ------------------------------------------------------------------
        data_log_files = [os.path.join(dir_name, elem) for elem in all_files_in_dir
                          if elem.endswith('.csv')]
        ret_data_logs[meth].append(data_log_files)
    for wkdir in work_dirs:
        local_wkdir = wkdir.split('/')[-1]
        meth = local_wkdir.split('_')[0]
        _process_files_in_curr_work_dir(wkdir, meth)
    env = first_env_as_list[0][1]
    experiment_info = first_info_as_list[0]
    return env, experiment_info, ret_data_logs, ret_results_from_files


def _get_workdirs_from_parent_directory_and_env_name(parent_dir, env_descr):
    """ Return a list of working directories. """
    dir_list = [elem for elem in os.listdir(parent_dir) if env_descr in elem]
    ret = [os.path.join(parent_dir, elem) for elem in dir_list]
    return ret


def plot_results(work_dirs, plot_from, profiled_info_dir, plot_order,
                 method_legend_colour_marker_dict, x_bounds=None, to_plot_legend=True,
                 plot_type='plot', options=None, **kwargs):
    """ Plots results.
        work_dirs is either a list of directories or a 2-tuple of the parent directory and
            an env_descr from which the list of directories can be obtained.
        results is a dictionary mapping the method to a list of files.
        plot_order is the order in which to plot methods.
        method_legend_colour_marker_dict: is a dictionary mapping each method to the plot
                                          information.
    """
    # pylint: disable=unused-variable
    options = get_plot_options(options, **kwargs)
    if isinstance(work_dirs, tuple) and len(work_dirs) == 2:
        work_dirs = _get_workdirs_from_parent_directory_and_env_name(work_dirs[0], work_dirs[1])
    print('Printing results from directories:\n%s', '\n'.join(work_dirs))
    env, experiment_info, data_log_files, in_run_results_from_files = \
        load_results_files_from_workdirs(work_dirs)
    if plot_from == 'inrun':
        gen_curves_from_in_run_results(
            in_run_results_from_files,
            plot_order,
            method_legend_colour_marker_dict,
            options=options,
            x_bounds=x_bounds,
            plot_type=plot_type,
            to_plot_legend=to_plot_legend,
        )
        plt.draw()
        plt.show()
    elif plot_from == 'logs':
        profiled_info_bank = ProfiledInfoBank(profiled_info_dir)
        env_metrics, env_table_metrics, leaf_metrics_for_each_method = \
            process_log_files_lists_for_all_methods(
            env, data_log_files, profiled_info_bank, experiment_info,
            method_order=plot_order, util_compute_grid_size=options['util_compute_grid_size'])
        create_bar_plots_from_leaf_metrics(
            leaf_metrics_for_each_method,
            plot_order,
            method_legend_colour_marker_dict,
            leafs_to_plot=list(env.leaf_nodes),
            options=options)
        _ = gen_curves_from_env_metrics(
            env_metrics,
            plot_order,
            ENV_CURVE_KEYS,
            method_legend_colour_marker_dict,
            options=options,
            x_bounds=x_bounds,
            plot_type=plot_type,
            to_plot_legend=to_plot_legend)
        # Print out the table metrics
        print(env_table_metrics)
        plt.draw()
        plt.show()
    else:
        raise ValueError('Unknown argumenet for plot_from: %s.'%(plot_from))


def plot_autoscaling_results(work_dirs, plot_from, profiled_info_dir, plot_order,
                             method_legend_colour_marker_dict, x_bounds=None, to_plot_legend=True,
                             plot_type='plot', options=None, **kwargs):
    """ Plots autoscaling results.
        See plot_results for descriptions of the arguments.
        work_dirs is either a list of directories or a 2-tuple of the parent directory and
            an env_descr from which the list of directories can be obtained.
        results is a dictionary mapping the method to a list of files.
        plot_order is the order in which to plot methods.
        method_legend_colour_marker_dict: is a dictionary mapping each method to the plot
                                          information.
    """
    options = get_plot_options(options, **kwargs)
    if isinstance(work_dirs, tuple) and len(work_dirs) == 2:
        work_dirs = _get_workdirs_from_parent_directory_and_env_name(work_dirs[0], work_dirs[1])
    print('Printing results from directories:\n%s', '\n'.join(work_dirs))
    env, experiment_info, data_log_files, _ = \
        load_results_files_from_workdirs(work_dirs)
    if plot_from == 'logs':
        profiled_info_bank = ProfiledInfoBank(profiled_info_dir)
        env_metrics, _, _ = \
            process_log_files_lists_for_all_methods(
            env, data_log_files, profiled_info_bank, experiment_info,
            method_order=plot_order, util_compute_grid_size=options['util_compute_grid_size'])
        # curves are generated here ------------------------------
        as_metrics = gen_auto_scaling_curves_from_env_metrics(
            env_metrics,
            plot_order,
            AUTOSCALING_CURVE_KEYS,
            method_legend_colour_marker_dict,
            options=options,
            x_bounds=x_bounds,
            plot_type=plot_type,
            to_plot_legend=to_plot_legend)
        # Print out the autoscaling metrics
        print()
        for key, val in as_metrics.items():
            print('leaf=%s'%(key))
            print(val)
        plt.draw()
        plt.show()
    else:
        raise ValueError('Unknown argumenet for plot_from: %s.'%(plot_from))

