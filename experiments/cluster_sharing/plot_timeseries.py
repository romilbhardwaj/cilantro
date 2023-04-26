"""
    Plots the allocations for a given policy.
    -- kirthevasank
"""

import argparse
from cilantro.ancillary.plot_with_time import plot_alloc_timeseries

WORKDIR = ('../../../results/apr16_2/', 'exp_env_1') # Experimental 1
PROFILED_INFO_DIR = 'profiled_oct7'

# POLICY_AND_FILENAME = ('mmflearn', 'mmflearn_exp_env_1_1000_eks_1008202325')
POLICY_NAME = 'mmflearn'
POLICY_NAME = 'multincadddec'

FIELD_LEGEND_MARKER_DICT = {
#     'minerva': {'colour': 'orange', 'linestyle': '-', 'legend': 'Minerva'},
#     'ernest': {'colour': 'yellow', 'linestyle': '-', 'legend': 'Ernest'},
#     'quasar': {'colour': 'brown', 'linestyle': '-', 'legend': 'Quasar'},
#     'parties': {'colour': 'silver', 'linestyle': '-', 'legend': 'Parties'},
#     'multincadddec': {'colour': 'maroon', 'linestyle': '-', 'legend': 'ADMI'},
#     'propfair': {'colour': 'black', 'linestyle': '-', 'legend': 'Resource-Fair'},
#     # Egal welfare
#     'evoegal': {'colour': 'olive', 'linestyle': '-', 'legend': 'EvoAlg-EW'},
#     'egalwelflearn': {'colour': 'lightgreen', 'linestyle': '-', 'legend': 'Cilantro-EW'},
#     # Util welfare
#     'evoutil': {'colour': 'purple', 'linestyle': '-', 'legend': 'EvoAlg-SW'},
#     'egal_welfare': {'colour': 'magenta', 'linestyle': '-', 'legend': 'Cilantro-SW'},
    'egal_welfare': {'colour': 'green', 'linestyle': '--', 'legend': 'Egalitarian welfare'},
    'util_welfare': {'colour': 'red', 'linestyle': '-.', 'legend': 'Social welfare'},
    # MMF
#     'sum_fairness_viol': {'colour': 'teal', 'linestyle': '-', 'legend': 'Sum fairness'},
#     'mean_fairness_viol': {'colour': 'cyan', 'linestyle': '-', 'legend': 'Mean fairness'},
    'max_fairness_viol': {'colour': 'blue', 'linestyle': '-', 'legend': 'NJC Fairness'},
    }



def main():
    """ Main function. """
    parser = argparse.ArgumentParser(description='Arguments for running plotting.')
    parser.add_argument('--plot-from', '-from', type=str, default='logs',
                        help='Should be inrun or logs. Specifies which data to plot.')
#     parser.add_argument('--profiled-info-dir', '-pid', type=str, default='',
#                         help='Directory which has the profiled data saved.')
    args = parser.parse_args()

    # Plot results ----------------------------------------------------------------
    save_fig_dir = WORKDIR[0]
    options = {'util_compute_grid_size': 1000,
               'plot_save_dir': save_fig_dir,
               'time_min': 10,
               'time_max': 6 * 60 * 60,
               'time_grid_size': 100,
                }
    plot_alloc_timeseries(WORKDIR, args.plot_from, POLICY_NAME,
                          PROFILED_INFO_DIR, save_fig_dir, FIELD_LEGEND_MARKER_DICT,
                          options=options)


if __name__ == '__main__':
    main()

