"""
    Some utilities for extracting metrics from raw data.
    -- kirthevasank
"""

import numpy as np


def latency_metrics_from_num_successes(num_events, num_successes):
    """ Compute latency metrics from number of successes.
        num_events is the total number of queries in any time window and num_successes is the
        number of queries completed under the latency deadline.
    """
    if num_events == 0:
        return None, None
    reward = num_successes/num_events
    sigma = 1/np.sqrt(2 * num_events)
    return reward, sigma


def latency_metrics_from_quantile_histogram(num_events, slo_latency, quantile_histogram,
                                            use_bernoulli_sigma=False):
    """ Compute latency metrics from a quantile histogram which gives the latency taken for each
        quantile.
        - quantile histogram is a list of 2-tuples of the form
          [(q1, l1), (q2, l2), ... (q(k-1), l(k-1), (qk, lk)]
          where q1 <= q2 <= ... <= q(k-1) <= qk. Here, qi is a number between 0 and 1 for all i and
          qk = 1 (i.e. the last quantile is always 1).
          l1 is the time taken to complete q1 fraction of the queries, l2 is the time taken for q2
          fraction of the queries etc.
        - num_events is the total number of events in the time window and slo_latency is the latency
          deadline for the SLO.
    """
    if isinstance(quantile_histogram, dict):
        quantiles = list(quantile_histogram)
        quantiles.sort()
        latencies = [quantile_histogram[q] for q in quantiles]
    else:
        quantiles = [elem[0] for elem in quantile_histogram]
        latencies = [elem[1] for elem in quantile_histogram]
    hist_size = len(quantiles)
    assert all(latencies[i] <= latencies[i+1] for i in range(hist_size-1))
    if slo_latency >= latencies[-1]:
        reward = 1.0
    elif slo_latency <= latencies[0]:
        reward = 0.0
    else:
        for idx, lat in enumerate(latencies):
            if lat >= slo_latency:
                rew_idx = idx - 1
                gap_frac = (slo_latency - latencies[rew_idx]) / (lat - latencies[rew_idx])
                break
        reward = quantiles[rew_idx] + gap_frac * (quantiles[rew_idx + 1] - quantiles[rew_idx])
    if use_bernoulli_sigma:
        sigma = 1/np.sqrt(2 * num_events)
    else:
        sigma = 0.5
    return reward, sigma


def latency_metrics_from_latency_histogram(latency_hist_bin_boundaries, latency_hist_heights,
                                           slo_latency):
    """ Computes the latency metrics from latency histogram which is a histogram where bins are
        latency values and the height is the number of events completed in that bin.
        - latency_hist_bin_boundaries is a list of size n+1 giving the boundaries of a histogram of
          size n.
        - latency_hist_heights are the heights of the histogram. Its (i)th element is the number of
          events that fall between the (i)th and (i+1)th values of latency_hist_bin_boundaries.
        - slo_latency is the latency deadline for the SLO.
    """
    hist_size = len(latency_hist_heights)
    if len(latency_hist_bin_boundaries) == hist_size:
        latency_hist_bin_boundaries = latency_hist_bin_boundaries[:] + [np.inf]
    assert len(latency_hist_bin_boundaries) == hist_size + 1 and \
           all(latency_hist_bin_boundaries[i] <= latency_hist_bin_boundaries[i+1]
                for i in range(len(hist_size)-1))
    cumulative_hist = np.cumsum(latency_hist_heights)
    num_events = sum(latency_hist_heights)
    if slo_latency >= latency_hist_bin_boundaries[-1]:
        num_succ_events = num_events
    elif slo_latency <= latency_hist_bin_boundaries[0]:
        num_succ_events = 0
    else:
        rew_idx = 1
        for idx in range(hist_size):
            if (latency_hist_bin_boundaries[idx] < slo_latency
                <= latency_hist_bin_boundaries[idx + 1]):
                rew_idx = idx
                if not np.isfinite(latency_hist_bin_boundaries[idx+1]):
                    gap_frac = 0.5
                else:
                    gap_frac = (slo_latency - latency_hist_bin_boundaries[idx]) / \
                               (latency_hist_bin_boundaries[idx+1] -
                                latency_hist_bin_boundaries[idx])
        num_succ_events = cumulative_hist[rew_idx] + \
                          gap_frac * (cumulative_hist[rew_idx + 1] - cumulative_hist[rew_idx])
    return latency_metrics_from_num_successes(num_events, num_succ_events)


def latency_metrics_from_e2e_latencies(e2e_latencies, slo_latency):
    """ Computes latency metrics from end to end latencies. """
    num_events = len(e2e_latencies)
    succ_events = [int(elem <= slo_latency) for elem in e2e_latencies]
    num_successes = sum(succ_events)
    return latency_metrics_from_num_successes(num_events, num_successes)

