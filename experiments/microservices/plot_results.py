"""
    Plotting script for MS scheduler.
    -- kirthevasank
"""

from cilantro.ancillary.ms_plotting import plot_ms_results

RESULTS_DIR = 'workdirs_eks'
ENV_DESCR = 'hotelres'
TO_PLOT_LEGEND = True
PLOT_TIME_BOUNDS = (0, 6 * 3600)


METHOD_ORDER = [
    'mssile',
    'ucbopt',
    'msevoopt',
    'propfair'
    ]



METHOD_LEGEND_MARKER_DICT = {
    'propfair': {'colour': 'black', 'linestyle': ':', 'legend': 'Resource-Fair', 'marker':'s'},
    'mssile': {'colour': 'red', 'linestyle': '-.', 'legend': 'Interleave-Explore', 'marker':'o'},
    'msevoopt': {'colour': 'darkgreen', 'linestyle': '--', 'legend': 'EvoAlg', 'marker':'^'},
    'ucbopt': {'colour': 'blue', 'linestyle': '-', 'legend': 'Cilantro', 'marker':'x'},
    }


def main():
    """ Main function. """
    save_fig_dir = RESULTS_DIR
    options = {'util_compute_grid_size': 1000,
               'plot_save_dir': save_fig_dir
#                'axis_font_size': 20, workdirs_eks/msile_hotelres_152_None_0418074408/hr-client.csv
                }
    plot_ms_results(results_dir=RESULTS_DIR,
                    method_order=METHOD_ORDER,
                    env_descr=ENV_DESCR,
                    x_bounds=PLOT_TIME_BOUNDS,
                    method_legend_colour_marker_dict=METHOD_LEGEND_MARKER_DICT,
                    to_plot_legend=TO_PLOT_LEGEND,
                    options=options,
                    )


if __name__ == '__main__':
    main()

