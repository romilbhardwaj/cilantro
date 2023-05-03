"""
    Plots the results for the experiments
    -- kirthevasank
"""

import argparse

from cilantro.ancillary.plotting import plot_results

WORK_DIRS = ('workdirs_eks', 'exp_env_1') # Experimental 1


METHOD_ORDER = [
    'mmf',
    'mmflearn',
    # 'utilwelforacle',
    'utilwelflearn',
    # 'egalwelforacle',
    # 'egalwelflearn',
    # 'propfair',
    # 'greedyegal',
    # 'evoutil',
    # 'evoegal',
    # 'ernest',
    # 'quasar',
    # 'minerva',
    # 'parties',
    # 'multincadddec',
    ]

METHOD_LEGEND_MARKER_DICT = {
    'minerva': {'colour': 'orange', 'linestyle': '-', 'legend': 'Minerva'},
    'ernest': {'colour': 'yellow', 'linestyle': '-', 'legend': 'Ernest'},
    'quasar': {'colour': 'brown', 'linestyle': '-', 'legend': 'Quasar'},
    'parties': {'colour': 'silver', 'linestyle': '-', 'legend': 'Parties'},
    'multincadddec': {'colour': 'maroon', 'linestyle': '-', 'legend': 'ADMI'},
    'propfair': {'colour': 'black', 'linestyle': '-', 'legend': 'Resource-Fair'},
    # Egal welfare
    'greedyegal': {'colour': 'teal', 'linestyle': '-', 'legend': 'Greedy-EW'},
    'evoegal': {'colour': 'olive', 'linestyle': '-', 'legend': 'EvoAlg-EW'},
    'egalwelflearn': {'colour': 'lightgreen', 'linestyle': '-', 'legend': 'Cilantro-EW'},
    'egalwelforacle': {'colour': 'darkgreen', 'linestyle': '--', 'legend': 'Oracle-EW'},
    # Util welfare
    'evoutil': {'colour': 'purple', 'linestyle': '-', 'legend': 'EvoAlg-SW'},
    'utilwelflearn': {'colour': 'magenta', 'linestyle': '-', 'legend': 'Cilantro-SW'},
    'utilwelforacle': {'colour': 'red', 'linestyle': '--', 'legend': 'Oracle-SW'},
    # MMF
    'mmflearn': {'colour': 'cyan', 'linestyle': '-', 'legend': 'Cilantro-NJC'},
    'mmf': {'colour': 'blue', 'linestyle': '--', 'legend': 'Oracle-NJC'},
    }


def main():
    """ Main function. """
    # parse args
    parser = argparse.ArgumentParser(description='Arguments for running plotting.')
    parser.add_argument('--plot-from', '-from', type=str, default='logs',
                        help='Should be inrun or logs. Specifies which data to plot.')
    parser.add_argument('--profiled-info-dir', '-pid', type=str, default='profiled_oct7',
                        help='Directory which has the profiled data saved.')
    args = parser.parse_args()

    # Plot results ----------------------------------------------------------------
    save_fig_dir = WORK_DIRS[0]
    options = {'util_compute_grid_size': 1000,
               'plot_save_dir': save_fig_dir
#                'axis_font_size': 20,
                }
    plot_results(WORK_DIRS, args.plot_from, args.profiled_info_dir,
                 METHOD_ORDER, METHOD_LEGEND_MARKER_DICT, save_fig_dir=save_fig_dir,
                 options=options)


if __name__ == '__main__':
    main()

