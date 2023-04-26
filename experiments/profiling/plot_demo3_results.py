"""
    Plots the results from the asynchronous demo.
    -- kirthevasank
"""

import argparse

from cilantro.ancillary.plotting import plot_results

# WORK_DIRS = [
#     'workdirs_kind/minerva_simple_12_kind_0916003250',
#     'workdirs_kind/minerva_simple_12_kind_0916003817',
#     'workdirs_kind/mmflearn_simple_12_kind_0916000455',
#     'workdirs_kind/mmflearn_simple_12_kind_0916001518',
#     'workdirs_kind/mmf_simple_12_kind_0915235059',
#     'workdirs_kind/mmf_simple_12_kind_0915235709',
#     'workdirs_kind/propfair_simple_12_kind_0916002029',
#     'workdirs_kind/propfair_simple_12_kind_0916002701',
#     ]
# METHOD_ORDER = ['propfair', 'minerva', 'mmflearn', 'mmf']

# WORK_DIRS = [
#     'workdirs_eks/minerva_simple_16_eks_0916032205',
#     'workdirs_eks/mmflearn_simple_16_eks_0916030856',
#     'workdirs_eks/mmf_simple_16_eks_0916040633',
#     'workdirs_eks/propfair_simple_16_eks_0916033319',
#     ]
# METHOD_ORDER = ['propfair', 'minerva', 'mmflearn', 'mmf']


WORK_DIRS = [
    'workdirs_eks/minerva_simple_16_eks_0916144952',
    'workdirs_eks/mmflearn_simple_16_eks_0916043921',
    ]
METHOD_ORDER = ['minerva', 'mmflearn']






METHOD_LEGEND_MARKER_DICT = {
    'propfair': {'colour': 'red', 'linestyle': '--', 'legend': 'Proportional Fairness'},
    'minerva': {'colour': 'green', 'linestyle': '-.', 'legend': 'Minerva'},
    'mmflearn': {'colour': 'blue', 'linestyle': '-', 'legend': 'Cilantro+MMF'},
    'mmf': {'colour': 'black', 'linestyle': '-', 'legend': 'MMF'},
    }


def main():
    """ Main function. """
    # parse args
    parser = argparse.ArgumentParser(description='Arguments for running plotting.')
    parser.add_argument('--plot-from', '-from', type=str, default='logs',
                        help='Should be inrun or logs. Specifies which data to plot.')
    parser.add_argument('--profiled-info-dir', '-pid', type=str, default='',
                        help='Directory which has the profiled data saved.')
    args = parser.parse_args()

    # Plot results ----------------------------------------------------------------
    plot_results(WORK_DIRS, args.plot_from, args.profiled_info_dir,
                 METHOD_ORDER, METHOD_LEGEND_MARKER_DICT)


if __name__ == '__main__':
    main()

