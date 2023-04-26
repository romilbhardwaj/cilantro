"""
    Plots autoscaling results.
    -- romilbhardwaj
    -- kirthevasank
"""

import argparse
# Local
from cilantro.ancillary.plotting_duplicate import plot_autoscaling_results
# from cilantro.ancillary.plotting import plot_autoscaling_results

# WORK_DIRS = ('../../../archive/as_eks_asds1_09251755/', 'asds1')
WORK_DIRS = ('../../../archive/as_eks_asim1_09281833', 'asim1')

METHOD_ORDER = ['k8sas', 'pidas', 'ds2', 'aslearn', 'asoracle']

METHOD_LEGEND_MARKER_DICT = {
    'k8sas': {'colour': 'green', 'linestyle': '-', 'legend': 'K8S-AS'},
    'pidas': {'colour': 'red', 'linestyle': '-', 'legend': 'PD'},
    'ds2': {'colour': 'orange', 'linestyle': '-', 'legend': 'DS2'},
    # Cilantro-based
    'aslearn': {'colour': 'cyan', 'linestyle': '-', 'legend': 'Cilantro-AS'},
    'asoracle': {'colour': 'blue', 'linestyle': '--', 'legend': 'Oracle-AS'},
    }


def main():
    """ main function. """
    # parse args
    parser = argparse.ArgumentParser(description='Arguments for running plotting.')
    parser.add_argument('--plot-from', '-from', type=str, default='logs',
                        help='Should be inrun or logs. Specifies which data to plot.')
    parser.add_argument('--profiled-info-dir', '-pid', type=str, default='',
                        help='Directory which has the profiled data saved.')
    args = parser.parse_args()

    # Plot results ----------------------------------------------------------------
    options = {'util_compute_grid_size': 1000,
                }
    save_fig_dir = WORK_DIRS[0]
    plot_autoscaling_results(WORK_DIRS, args.plot_from, args.profiled_info_dir,
                             METHOD_ORDER, METHOD_LEGEND_MARKER_DICT, save_fig_dir=save_fig_dir,
                             options=options)


if __name__ == '__main__':
    main()

