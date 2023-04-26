"""
    Implements some important utilities for fair allocation.
    -- romilbhardwaj
    -- kirthevasank
"""

# pylint: disable=invalid-name

import numpy as np


def resource_loss(demands, allocs, resource_quantity, to_normalise=True):
    """ The amount of resources left on the table. """
    if (demands is None) or (demands[0] is None):
        return None # Usually because of unknown demands. Return None
    demands = np.array(demands)
    allocs = np.array(allocs)
    unallocated_resources = resource_quantity - allocs.sum()
    overallocated_resources = np.clip(allocs - demands, 0, np.inf).sum()
    unmet_demand = np.clip(demands - allocs, 0, np.inf).sum()
    loss = min(unallocated_resources + overallocated_resources, unmet_demand)
    if to_normalise:
        loss /= resource_quantity
    return loss


def fairness_violation(demands, allocs, fair_share, resource_quantity, alloc_granularity=None):
    """ Compute fairness violation. """
    if (demands is None) or (demands[0] is None):
        return None # Usually because of unknown demands. Return None
    demands = np.array(demands)
    allocs = np.array(allocs)
    if alloc_granularity is not None:
        fair_share = [np.floor(elem / alloc_granularity) * alloc_granularity for elem in fair_share]
    fair_share = np.array(fair_share)
    min_dem_fs = np.minimum(demands, fair_share)
    viols = np.maximum(0, min_dem_fs - allocs)
    norm_viols = viols / fair_share
    avg_viol = np.mean(norm_viols)
    max_viol = np.max(norm_viols)
    sum_viol = np.sum(viols / resource_quantity)
    return sum_viol, avg_viol, max_viol


def useful_resource_fraction(demands, allocs, resource_quantity, to_normalise=True):
    """ Computes the useful utilisation for the given allocation. """
    demands = np.array(demands)
    allocs = np.array(allocs)
    total_resource_used = np.minimum(demands, allocs).sum()
    if to_normalise:
        return total_resource_used / resource_quantity
    else:
        return total_resource_used


def utilitarian_welfare(utils, to_normalise=True):
    """ Computes utilitarian welfare. """
    if to_normalise:
        return np.mean(utils)
    else:
        return np.sum(utils)


def egalitarian_welfare(utils):
    """ Egalitarian welfare. """
    return np.min(utils)

