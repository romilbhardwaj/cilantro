"""
    Process profiling data.
    -- kirthevasank
    -- romilbhardwaj
"""

# pylint: disable=too-many-locals

import argparse
import logging
import os
import pickle
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#Local
from cilantro.ancillary.info_write_load_utils import read_experiment_info_from_files
from cilantro.ancillary.plotting import COLOURS, transparent
from cilantro.core.henv import are_two_environments_equal
from cilantro.learners.ibtree import IntervalBinaryTree
from cilantro.learners.binning_est import BinningEst
from cilantro.profiling.profiled_info_loader import ProfiledInfoBank


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DFLT_PROFILE_RESULTS_SAVE_DIR = 'results_profiling'
DFLT_GRID_SIZE = 200

ALLOC_PER_UNIT_LOAD = 'alloc_per_unit_load'
REWARD = 'reward'
TIME_PERIOD = 'time_period'
SIGMA = 'sigma'
LOAD = 'load'

# EST_SIGMA_FROM_BINNING = True
EST_SIGMA_FROM_BINNING = False

NUM_INIT_ROWS_TO_IGNORE = 10



def process_data_log_file(file_name):
    """ Process file. """
#     logger.info('Loading data from %s'%(file_name))
    df = pd.read_csv(file_name, index_col=0)
    ret = []
    row_idx = 0
    for _, row in df.iterrows():
        row_idx += 1
        if row_idx <= NUM_INIT_ROWS_TO_IGNORE:
#             logger.info('ignoring data on row_idx %d,  %s', row_idx, str(row))
            continue
        curr_data = {TIME_PERIOD: row['event_end_time'] - row['event_start_time'],
                     REWARD: row['reward'],
                     ALLOC_PER_UNIT_LOAD: row['alloc']/row['load'],
                     SIGMA: row['sigma'],
                     LOAD: row['load']}
        ret.append(curr_data)
    return ret


def process_data_from_dir(dir_name):
    """ Process data from dir. """
    try:
        env, _ = read_experiment_info_from_files(dir_name)
    except FileNotFoundError:
        return None, None
    csv_files = [elem for elem  in os.listdir(dir_name) if elem.endswith('.csv')]
    ret = {}
    for csvf in csv_files:
        csv_file_path = os.path.join(dir_name, csvf)
        descr = csvf.split('.')[0]
        workload_descr = descr.split('_')[1]
        if not (workload_descr in ret):
            ret[workload_descr] = []
        ret[workload_descr].extend(process_data_log_file(csv_file_path))
    return env, ret


def process_data_from_list_of_dirs(list_of_dirs):
    """ Processes data from a list of dirs. """
    ret = {}
    reference_env = None
    for csv_dir in list_of_dirs:
        env, curr_dir_data = process_data_from_dir(csv_dir)
        if env is None:
            continue
        # First cross-check the environment -----------------------------------------------------
        if reference_env:
            assert are_two_environments_equal(reference_env, env)
        else:
            reference_env = env
        for key, val in curr_dir_data.items():
            if not key in ret:
                ret[key] = []
            ret[key].extend(val)
#     for _, val in ret.items():
#         random.shuffle(val)
    return ret


def get_estimates_for_workload_type(data, workload_type, grid=None, grid_size=None):
    """ Returns means, ucbs and lcbs for the workload type. """
    X = [elem[ALLOC_PER_UNIT_LOAD] for elem in data]
    Y = [elem[REWARD] for elem in data]
    Sigmas = [elem[SIGMA] for elem in data]
    loads = [elem[LOAD] for elem in data]
    int_lb = min(X)
    int_ub = max(X)
    min_y = min(Y)
    max_y = max(Y)
    if 'mltrain' in workload_type:
        glob_lower_bound = 0.0
        glob_upper_bound = 2.0 * max_y
    else:
        glob_lower_bound = 0.0
        glob_upper_bound = 1.0
    logger.info('global bounds: %s', (glob_lower_bound, glob_upper_bound))
    lip_const = 8 * (max_y - min_y)/(int_ub - int_lb)
    p50_load = np.percentile(loads, 50)
    p99_load = np.percentile(loads, 99)
    # Create a binning model to estimate sigma ---------------------------------
    if EST_SIGMA_FROM_BINNING:
        bin_model = BinningEst(workload_type, int_lb, int_ub, 100)
        bin_model.initialise_model()
        bin_model.add_multiple_data_points(X, Y, None)
        est_sigma = 1 * bin_model.get_sigma_estimate()
        logger.info('sigma estimate for workload_type %s is %0.4f', workload_type, est_sigma)
#         import pdb; pdb.set_trace()
        Sigmas = [est_sigma] * len(Sigmas)
    est_model = IntervalBinaryTree(workload_type, int_lb, int_ub, lip_const,
                                   glob_lower_bound=glob_lower_bound,
                                   glob_upper_bound=glob_upper_bound)
    est_model = BinningEst(workload_type, int_lb, int_ub, glob_lower_bound, glob_upper_bound, 30)
    est_model.initialise_model()
    est_model.add_multiple_data_points(X, Y, Sigmas)
#     print(np.mean(Sigmas))
#     print('max_height_expanded, num_data_in_tree',
#           est_model.max_height_expanded, est_model.num_data_in_tree)
#     import pdb; pdb.set_trace()
    # Now obtain estimates -------------------------------------
    if grid is None:
        grid = np.linspace(int_lb, int_ub, grid_size)
    ret_ests = []
    ret_lcbs = []
    ret_ucbs = []
    ret_lcbs_dist = []
    ret_ucbs_dist = []
    for test_pt in grid:
        est, lcb, ucb = est_model.compute_estimate_for_input(test_pt)
        _, lcb_dist, ucb_dist = est_model.compute_dist_estimate_for_input(test_pt)
        ret_ests.append(est)
        ret_lcbs.append(lcb)
        ret_ucbs.append(ucb)
        ret_lcbs_dist.append(lcb_dist)
        ret_ucbs_dist.append(ucb_dist)
    ret = {'grid': grid, 'lcbs': ret_lcbs, 'ucbs': ret_ucbs, 'ests': ret_ests,
           'workload_type': workload_type, 'lip_const': lip_const, 'int_ub':int_ub,
           'p50_load': p50_load, 'p99_load': p99_load,
           'ucbs_dist': ret_ucbs_dist, 'lcbs_dist': ret_lcbs_dist}
    return ret


def plot_profiled_data(est_dict, data=None, test_point=None, save_descr=None):
    """ Plot profiled data. """
    plt.figure(figsize=(16, 12))
    if data:
        X = [elem[ALLOC_PER_UNIT_LOAD] for elem in data]
        Y = [elem[REWARD] for elem in data]
        plt.scatter(X, Y, marker='x', linewidth=5, color=COLOURS['grey'])
    plt.fill_between(
        est_dict['grid'], est_dict['lcbs_dist'], est_dict['ucbs_dist'],
        color=transparent(*COLOURS['green'], opacity=0.2))
    plt.fill_between(
        est_dict['grid'], est_dict['lcbs'], est_dict['ucbs'],
        color=transparent(*COLOURS['aqua'], opacity=0.5))
    plt.plot(est_dict['grid'], est_dict['ests'], color=COLOURS['blue'], linewidth=3)
    if test_point:
        X = test_point[0]
        Y = test_point[1]
        plt.scatter([X], [Y], marker='x', linewidth=10, color='r')
        plt.plot([0, X], [Y, Y], linestyle='--', linewidth=3, color='r')
        plt.plot([X, X], [0, Y], linestyle='--', linewidth=3, color='r')
    plt.title(est_dict['workload_type'], fontsize=12)
    # Pretty up figures ----------------------------------------------------
    # ---------------------------------------------------
    if save_descr:
        plt.savefig(save_descr + '.pdf', format='pdf')


def _get_data_log_dirs_from_file(logs_dir, logs_dirs_file):
    """ Returns a list of directory names from the file. """
    if logs_dir != '':
        dir_names = list(os.listdir(logs_dir))
        ret = [os.path.join(logs_dir, elem) for elem in dir_names]
        ret.sort()
    else:
        with open(logs_dirs_file, 'r') as file_handle:
            ret = [elem.strip() for elem in file_handle.readlines()]
            ret = [elem for elem in ret if elem != '']
            file_handle.close()
    return ret


def main():
    """ Main function. """
    parser = argparse.ArgumentParser(description='Arguments for processing profiled data.')
    parser.add_argument('--logs-dir', '-logsdir', type=str, default='',
                        help='The directory which contains the logged data.')
    parser.add_argument('--logs-dirs-file', '-logsfile', type=str, default='',
                        help='A file which consists of all the directories storing logged data.')
    parser.add_argument('--save-dir', '-sd', type=str, default='',
                        help='Directory to save the profiled results.')
    parser.add_argument('--grid-size', '-gs', type=int, default=DFLT_GRID_SIZE,
                        help='Grid size for profiling.')
    parser.add_argument('--to-plot', '-plot', type=int, default=0,
                        help='Plots the results if true.')
    args = parser.parse_args()

    save_dir = args.save_dir
    data_log_dirs = _get_data_log_dirs_from_file(args.logs_dir, args.logs_dirs_file)
    logger.info('Profiling results from: %s.', ', '.join(data_log_dirs))
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    all_data_from_dirs = process_data_from_list_of_dirs(data_log_dirs)
    # Process and save the data ---------------------------------------------------------------
    for workload_type, wl_data in all_data_from_dirs.items():
        wl_est_dict = get_estimates_for_workload_type(wl_data, workload_type,
                                                      grid_size=args.grid_size)
        if save_dir:
            wl_pickle_file_name = os.path.join(save_dir, workload_type + '.p')
            with open(wl_pickle_file_name, 'wb') as wl_save_file:
                pickle.dump(wl_est_dict, wl_save_file)
                wl_save_file.close()
            logger.info('Saved workload_type:%s data to %s.', workload_type, wl_pickle_file_name)
        if args.to_plot:
            plot_profiled_data(wl_est_dict, wl_data)

    # Load the saved data and plot it ---------------------------------------------------------
    profiled_info_bank = ProfiledInfoBank(save_dir)
    for wl_type, profiled_info in profiled_info_bank.enumerate():
        threshold = (0.1 + np.random.random() * 0.9) * max(profiled_info.profiled_data['ests'])
        unit_demand = profiled_info.get_alloc_for_payoff_vals([threshold])[0]
        logger.info('Estimated unit demand for workload_type:%s for threshold:%0.4f = %0.4f',
                    wl_type, threshold, unit_demand)
        plot_profiled_data(profiled_info.profiled_data, data=None,
                           test_point=(unit_demand, threshold))

    if args.to_plot:
        plt.show()


if __name__ == '__main__':
    main()

