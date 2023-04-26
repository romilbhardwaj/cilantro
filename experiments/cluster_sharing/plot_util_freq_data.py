"""
    Plots utility frequency data.
    -- kirthevasank
"""

# WORK_DIRS = ('../../../results/apr16_2/', 'exp_env_1') # Experimental 1



import numpy as np
import matplotlib.pyplot as plt

FIGSIZE = (32, 12)
XTICK_FONTSIZE = 20
YTICK_FONTSIZE = 40
LEGEND_FONT_SIZE = 35
BAR_ANNOT_FONT_SIZE = 60
AXIS_FONT_SIZE = 90

BAR_WIDTH = 0.7


def autolabel(rect, label, ax, font_col):
    """Attach a text label above each bar in *rects*, displaying its height."""
    height = np.round(rect.get_height(), 1)
    label = np.round(label, 1)
    ax.annotate('%0.1f'%(label),
                xy=(rect.get_x() + rect.get_width() / 2, height * 0.1),
                xytext=(0, 0),  # 3 points vertical offset
                textcoords="offset points", color=font_col,
                ha='center', va='bottom', fontsize=BAR_ANNOT_FONT_SIZE,
                rotation=90)



def main():
    """ main. """
    bar_vals = {
        'db01': [3318, 3076, 3445, 2895, 3007, 3276, 4103, 3355, 3581, 3493, 3202, 3378, 3368, 3803],
        'db15': [2887, 3011, 3191, 2774, 2878, 2883, 3334, 3046, 3491, 3236, 2925, 3657, 3149, 3428],
        'mlt4': [1065,  865,  801, 1084, 1273, 1050,  714, 1367, 1006,  815,  907, 1361, 1225, 1080],
        'prs3': [1822, 2880, 2850, 2457, 2357, 2413, 2698, 3402, 2411, 2458, 2291, 2427, 2194, 2929],
        }
    fig, ax = plt.subplots(figsize=(16, 9))
    time_est = 14 * 10
    users = list(bar_vals)
    bar_ys = [sum(bar_vals[elem])/time_est for elem in users]
    num_users = len(users)

    all_rects = [None] * num_users
#     X = np.arange(num_users)
    for idx, usr in enumerate(users):
        bar_mean = bar_ys[idx]
        all_rects[idx] = ax.bar(idx, bar_mean, BAR_WIDTH, color='c')
    ax.set_ylabel('Num updates per hour', fontsize=50)
    user_labels = [''] + users
    ax.set_xticks(np.arange(-1, len(user_labels)-1, step=1))
    ax.set_xticklabels(user_labels, fontsize=50)
    ax.set_xlim([-0.5, 3.5])
    ax.tick_params(axis='y', labelsize=40)

    for idx, rect in enumerate(all_rects):
#         print('idx', idx)
#         print(legends[idx], cols[idx])
#         bar_label, bar_col = legends[idx], cols[idx]
#         if bar_col in ['b', 'k']:
#             font_col = 'white'
#         else:
#             font_col = 'k'
        autolabel(rect[0], bar_ys[idx], ax, 'k')


#     ax = fig.add_axes([0,0,1,1])
#     ax.bar(bar_xs, bar_ys)
    fig.tight_layout()
    fig.savefig('update_freq.pdf', format='pdf')
    
    plt.show()



if __name__ == '__main__':
    main()

