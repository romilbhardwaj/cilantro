"""
    A simple demo using time series.
    -- kirthevasank
"""

# pylint: disable=too-many-locals


import time
import pickle
import numpy as np
from matplotlib import pyplot as plt
# Local
from cilantro.timeseries.arima import ARIMATSModel


# DATA_PATH = '/home/kirthevasan/projects/Autosc/data/twit-b1000-n88600.p'
DATA_PATH = '/home/kirthevasan/projects/Autosc/data/twit-b60000-n1476.p'


arima_orders = [
#     (0, 0, 1),
    (1, 1, 1),
#     (5, 1, 0),
#     (1, 1, 2),
#     (2, 1, 2),
#     (0, 0, 5),
#     (3, 0, 2),
    ]


def plot_ts(orig_time_series):
    """ Plots time series. """
    num_orig_data = len(orig_time_series)
    plt.figure()
    plt.plot(range(num_orig_data), orig_time_series)
    plt.show()


def main():
    """ Main function. """
    with open(DATA_PATH, 'rb') as data_file:
        data_dict = pickle.load(data_file)
    orig_time_series = [elem[0] for elem in data_dict['data']]
    len_time_series = len(orig_time_series)

    conf_alpha = 0.90
#     start_at = 50
#     max_num_tests = 20
    start_at = 1200
    max_num_tests = 200
#     start_at = 1000
#     max_num_tests = 200
    errors = {}
    trapped_by_ci = {}
    avg_times = {}
    conf_widths = {}
    # Create model
    for ao in arima_orders:
        curr_errs = []
        curr_trapped_by_ci = []
        curr_conf_widths = []
        curr_times = []
        print('Model %s'%(str(ao)))
        for pred_idx in range(start_at + 1, min(start_at + 1 + max_num_tests, len_time_series - 1)):
            start_time = time.time()
            model = ARIMATSModel(str(ao), ao)
            model.initialise_model()
            history = orig_time_series[:pred_idx]
            model.update_model_with_new_data(history)
            mean_pred, lcb, ucb = model.forecast(num_steps_ahead=1, conf_alpha=conf_alpha)
            tot_time = time.time() - start_time
            true_val = orig_time_series[pred_idx]
            err = abs((true_val - mean_pred) / true_val)
            is_in_conf = lcb <= true_val <= ucb
            curr_errs.append(err)
            curr_trapped_by_ci.append(is_in_conf)
            curr_times.append(tot_time)
            curr_conf_widths.append((ucb-lcb)/true_val)
        errors[ao] = np.mean(curr_errs)
        trapped_by_ci[ao] = np.mean(curr_trapped_by_ci)
        avg_times[ao] = np.mean(curr_times)
        conf_widths[ao] = np.mean(curr_conf_widths)
    print('errors', errors)
    print('trapped', trapped_by_ci)
    print('conf_widths', conf_widths)
    print('times', avg_times)


if __name__ == '__main__':
    main()

