"""
    Implements an evolutionary algorithm for optimisation.
    -- kirthevasank
    -- romilbhardwaj
"""


import logging
from copy import copy
import numpy as np
# Local


logger = logging.getLogger(__name__)


def get_initial_allocations(leafs, resource_quantity, num_initial_allocations,
                            min_alloc_per_leaf=1):
    """ Returns a set of initial allocations. """
    assert num_initial_allocations >= 1
    num_leafs = len(leafs)
    fair_alloc_list = list(resource_quantity * (np.ones((num_leafs, )) / num_leafs))
    ret_list = [fair_alloc_list]
    for _ in range(1, num_initial_allocations):
        unnorm_ratios = np.random.random((num_leafs, ))
        norm_ratios = unnorm_ratios / unnorm_ratios.sum()
        ret_list.append(list(min_alloc_per_leaf +
            (resource_quantity - min_alloc_per_leaf * num_leafs) * norm_ratios))
    ret = [dict(zip(leafs, elem)) for elem in ret_list]
    return ret


def get_dflt_mutation_op(min_num_steps, max_num_steps, min_alloc_per_leaf):
    """ Obtain the default mutation operator. """
    ret = lambda alloc: _dflt_mutation_op(
        alloc, min_num_steps=min_num_steps, max_num_steps=max_num_steps,
        min_alloc_per_leaf=min_alloc_per_leaf)
    return ret


def _dflt_mutation_op(curr_alloc, *args, **kwargs):
    """ Default mutation operator. """
    choose_probs = [0.25, 0.75]
    cum_probs = np.cumsum(choose_probs)
    chooser = np.random.random()
    if chooser <= cum_probs[0]:
        return _dflt_mutation_op_1(curr_alloc, *args, **kwargs)
    elif chooser <= cum_probs[1]:
        return _dflt_mutation_op_2(curr_alloc, *args, **kwargs)
    else: # redundant, but pylint is complaining
        return _dflt_mutation_op_2(curr_alloc, *args, **kwargs)


def _dflt_mutation_op_1(curr_alloc, min_num_steps=1, max_num_steps=5, min_alloc_per_leaf=1,
                        *args, **kwargs):
    """ Default mutation operation for the evolutionary algorithm. """
    # pylint: disable=unused-argument
    def _simple_step(_alloc, _amnt_to_flip=1):
        """ Takes and allocation and flips one resource from one agent to another. """
        keys_to_flip_from = [key for key, val in _alloc.items()
                             if val >= min_alloc_per_leaf + _amnt_to_flip]
        key_to_flip_from = np.random.choice(keys_to_flip_from)
        keys_to_flip_to = [key for key in _alloc if key != key_to_flip_from]
        key_to_flip_to = np.random.choice(keys_to_flip_to)
        _alloc[key_to_flip_from] -= _amnt_to_flip
        _alloc[key_to_flip_to] += _amnt_to_flip
    # Repeat _simple_step for max_num_steps times.
    num_steps = min_num_steps + np.random.randint(max_num_steps - min_num_steps) + 1
    ret = dict(curr_alloc.items())
    for _ in range(num_steps):
        _simple_step(ret)
    return ret


def _dflt_mutation_op_2(curr_alloc, perturb_factor=0.1, min_alloc_per_leaf=1, *args, **kwargs):
    """ 2nd Default mutation operator for the evolutionary algorithm. """
    # pylint: disable=unused-argument
    num_leafs = len(curr_alloc)
    leafs = list(curr_alloc)
    # Compute ratios without minimum allocation ------------------------------------------------
    orig_vals = [max(0, curr_alloc[key] - min_alloc_per_leaf) for key in leafs]
    sum_orig_vals = sum(orig_vals)
    norm_orig_vals = [elem/sum_orig_vals for elem in orig_vals]
    perturbations = list(perturb_factor * np.random.normal(size=(num_leafs,)))
    new_unnorm_allocs = [max(0, x+y) for x, y in zip(norm_orig_vals, perturbations)]
    sum_new_allocs = sum(new_unnorm_allocs)
    new_allocs = [min_alloc_per_leaf + (x * sum_orig_vals)/sum_new_allocs
                  for x in new_unnorm_allocs]
    new_allocs = dict(zip(leafs, new_allocs))
#     logger.info('leafs: %s', leafs)
#     logger.info('new_unnorm_allocs: %s', new_unnorm_allocs)
#     logger.info('perturbations: %s', perturbations)
#     logger.info('orig_vals: %s', orig_vals)
#     logger.info('curr_alloc: %s\nnew_allocs: %s', curr_alloc, new_allocs)
    return new_allocs


class EvoOpt:
    """ An evolutionary algorithm for optimising a function over the set of allocation. """

    def __init__(self, env, resource_quantity, mutation_op=None, num_mutations_per_epoch=5):
        """ Constructor.
            env, resource_quantity: the environment and resource quanttity.
            mutation_op: the mutation operator
            init_allocs_and_vals: a list of 2-tuple (alloc, val) pairs
        """
        self.env = env
        self.resource_quantity = resource_quantity
        if mutation_op:
            self._mutation_op = mutation_op
        else:
            num_leaf_nodes = len(self.env.leaf_nodes)
            max_num_steps_per_mutation = max(10, int(1.5 * num_leaf_nodes))
            self._mutation_op = lambda alloc: _dflt_mutation_op(
                alloc, min_num_steps=1, max_num_steps=max_num_steps_per_mutation)
        # Initialise the algorithm ------------------------------------------
        self._to_evaluate_queue = []
        self._history_allocs = []
        self._history_vals = []
        self.num_mutations_per_epoch = num_mutations_per_epoch
        self._curr_opt_val = -np.inf
        self._curr_opt_alloc = None

    def add_data(self, X, Y):
        """ Add data to the history. """
        self._history_allocs.extend(X)
        self._history_vals.extend(Y)
        for (x, y) in zip(X, Y):
            if y > self._curr_opt_val:
                self._curr_opt_val = y
                self._curr_opt_alloc = x

    def get_curr_optimum(self):
        """ Returns the current optimum. """
        ret = (copy(self._curr_opt_alloc), self._curr_opt_val)
        return ret

    def generate_new_eval_points(self):
        """ Generates new points for evaluation and adds it to the queue. """
        sampled_idxs = _sample_according_to_exp_probs(self._history_vals,
                                                      self.num_mutations_per_epoch)
        past_allocs_to_mutate_from = [self._history_allocs[elem] for elem in sampled_idxs]
        new_mutations = [self._mutation_op(elem) for elem in past_allocs_to_mutate_from]
        self._to_evaluate_queue.extend(new_mutations)

    def get_next_eval_point(self):
        """ Returns the next point for evaluation. """
        if len(self._to_evaluate_queue) == 0:
            self.generate_new_eval_points()
        ret = self._to_evaluate_queue.pop(0)
        return ret


def _sample_according_to_exp_probs(vals, num_samples):
    """ Samples according to exponential probs. """
    if len(vals) == 1:
        return [0] * num_samples
    num_vals = len(vals)
    to_replace_when_sampling = (num_vals >= num_samples)
    vals = np.array(vals)
    std_val = vals.std()
    if std_val == 0:
        # Return random indices
        return np.random.choice(num_vals, size=(num_samples,), replace=to_replace_when_sampling)
    mean_val = vals.mean()
    norm_vals = (vals - mean_val) / std_val
    exp_vals = np.exp(norm_vals)
    prob_vals = exp_vals / exp_vals.sum()
    logger.info('\nnum_vals: %s.\n prob_vals: %s.\n num_samples: %s.\n vals: %s.\n')
    sampled_idxs = np.random.choice(num_vals, size=(num_samples,), p=prob_vals,
                                    replace=to_replace_when_sampling)
    return sampled_idxs


def optimise_with_evo_alg(func, env, resource_quantity, num_iters, prev_allocs_and_vals=None,
                          num_initial_allocations=None, min_alloc_per_leaf=1, *args, **kwargs):
    """ Optimises with an evolutionary algorithm. """
    # Create mutation operation and create an EvoOpt object ------------------------------------
    mutation_op = lambda alloc: _dflt_mutation_op(alloc, min_num_steps=1, max_num_steps=5,
                                                  min_alloc_per_leaf=min_alloc_per_leaf)
    evo_opt = EvoOpt(env, resource_quantity, mutation_op=mutation_op, *args, **kwargs)
    # Any previous data ------------------------------------------------------------------------
    if prev_allocs_and_vals:
        prev_allocs = [elem[0] for elem in prev_allocs_and_vals]
        prev_vals = [elem[1] for elem in prev_allocs_and_vals]
    else:
        prev_allocs = []
        prev_vals = []
    # Initialisation ---------------------------------------------------------------------------
    env_leafs = list(env.leaf_nodes)
    num_leafs = len(env_leafs)
    num_initial_allocations = num_initial_allocations if num_initial_allocations else \
                              max(10, 2 * num_leafs)
    init_allocs = get_initial_allocations(env_leafs, resource_quantity, num_initial_allocations)
    init_vals = [func(elem) for elem in init_allocs]
    evo_opt.add_data(prev_allocs + init_allocs, prev_vals + init_vals)
    # Proceed for num iters ------------------------------------------------------------------------
    for _ in range(num_iters):
        next_eval_pt = evo_opt.get_next_eval_point()
        next_val = func(next_eval_pt)
        evo_opt.add_data([next_eval_pt], [next_val])
    # Return -------------------------------------------------------------------------------
    return evo_opt.get_curr_optimum()

