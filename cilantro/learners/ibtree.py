"""
    Utilities in estimating and constructing confidence intervals for a nonparametric function.
    -- kirthevasank
    -- romilbhardwaj
"""

# In this module, generally speaking, we will use h to denote the height of a node in the binary
# tree and k to denote it's lateral position among nodes at height h.


import bisect
import logging
import numpy as np
# Local
from cilantro.learners.base_learner import LearningModel


logger = logging.getLogger(__name__)

MIN_NUM_DATA_IN_NODE_BEFORE_EXPANDING = 3

TAU_TUNE_COEFF = 1e-5
BETA_TUNE_COEFF = 1e-3

def get_left_child_h_k(h, k):
    """ Returns the h, k values for the left child. """
    return (h+1, 2*k-1)


def get_right_child_h_k(h, k):
    """ Returns the h, k values for the left child. """
    return (h+1, 2*k)


def get_default_delta():
    """ Return default delta. """
    return 0.001


def get_default_beta_t_fn(delta):
    """ Obtain default beta_t function. """
#     print('Setting tune_coeff to %0.4f'%(tune_coeff))
    beta_t = lambda t: BETA_TUNE_COEFF * np.sqrt(
        (4 + 2 * np.log(2)) * np.log(np.pi ** 2 * (2 ** np.ceil(np.log2(t + 1))) / (6 * delta))
        )
    return beta_t

# def get_default_tau_ht_fn(beta_t, lip_const_val, interval_width):
#     """ Obtain the default tau_ht function. """
#     def _tau_ht(h, t, _beta_t, _lip_const_val, _interval_width):
#         """ tau_ht. """
#         return max(3, (_beta_t(t) / (_lip_const_val/_interval_width))**2  * (4.0**h))
#     return lambda h, t: _tau_ht(h, t, beta_t, lip_const_val, interval_width)

# def get_default_tau_ht_fn(beta_t, lip_const_val, y_range):
#     """ Obtain the default tau_ht function. """
#     print('Received lip_const, y_range', lip_const_val, y_range)
#     def _tau_ht(h, t, _beta_t, _lip_const_val, _y_range):
#         """ tau_ht. """
#         return max(0.0, (_beta_t(t) / _lip_const_val)**2  * (4.0**h))
#     return lambda h, t: _tau_ht(h, t, beta_t, lip_const_val, y_range)

def get_default_tau_ht_fn(beta_t, lip_const_val):
    """ Obtain the default tau_ht function. """
    def _tau_ht(h, t, _beta_t, _lip_const_val):
        """ tau_ht. """
        return max(0.0, TAU_TUNE_COEFF * (_beta_t(t) / _lip_const_val)**2  * (4.0**h))
    return lambda h, t: _tau_ht(h, t, beta_t, lip_const_val)


class IntervalBinaryTreeNode:
    """ Node in an interval binary tree. """

    def __init__(self, h, k, parent, is_left_child, lip_const, bound_Blb, bound_Bub,
                 global_upper_bound, global_lower_bound):
        """ Constructor. """
        self.h = h
        self.k = k
        self.parent = parent
        self.is_left_child = is_left_child
        self.left_child = None
        self.right_child = None
        self.lip_const = lip_const
        # Create placeholders for the data under each node
        self.data_y_vals_for_node = []
        self.data_x_vals_for_node = []
        self.data_inv_sigmasq_vals_for_node = []
        self.num_data_in_node = 0
        # Create placeholders for B/f/g values
        self.Bub = bound_Bub
        self.Blb = bound_Blb
        self.fub = global_upper_bound
        self.flb = global_lower_bound
        self.sum_of_SG_consts = 0  # maintains sum of variances

    def get_B_val(self, perf_goal):
        """ Compute the B value. """
        return min(self.Bub - perf_goal, perf_goal - self.Blb)

    def is_a_leaf(self):
        """ Returns true if a leaf. """
        return (self.left_child is None) and (self.right_child is None)

    def add_data_point(self, y_val, sigma, x_val=None):
        """ Add data point. """
        self.data_y_vals_for_node.append(y_val)
        self.data_inv_sigmasq_vals_for_node.append(1/sigma**2)
        self.data_x_vals_for_node.append(x_val)
        self.num_data_in_node += 1

    def set_left_child(self, node):
        """ Sets left child. """
        self.left_child = node

    def set_right_child(self, node):
        """ Sets right child. """
        self.right_child = node


class IntervalBinaryTree(LearningModel):
    """ A binary tree on an interval. """

    def __init__(self, name, int_lb, int_ub, lip_const,
                 glob_lower_bound=-np.inf, glob_upper_bound=np.inf,
                 delta=None, beta_t=None, tau_ht=None):
        """ int_lb and int_ub are the lower and upper bounds of the interval.
        """
        super().__init__(name, int_lb, int_ub)
        self.int_width = self.int_ub - self.int_lb
        self.lip_const = lip_const
        self.root = None
        self.glob_lower_bound = glob_lower_bound
        self.glob_upper_bound = glob_upper_bound
        assert self.glob_lower_bound <= self.glob_upper_bound
        # Maintain a list of nodes at each height
        self.num_data_in_tree = None
        self.nodes_expanded_at_each_height = {}
        self.max_height_expanded = -1
        self.delta = delta if delta else get_default_delta()
        self.beta_t = beta_t if beta_t else get_default_beta_t_fn(self.delta)
#         y_range = glob_upper_bound - glob_lower_bound
        self.tau_ht = tau_ht if tau_ht else get_default_tau_ht_fn(self.beta_t, self.lip_const)

    def set_beta_t(self, beta_t):
        """ Sets beta_t """
        logger.info('beta_t was set externally')
        self.beta_t = beta_t

    def set_tau_ht(self, tau_ht):
        """ Sets tau_ht """
        self.tau_ht = tau_ht

    def _initialise_model_child(self):
        """ Creates the initial tree. """
        self.num_data_in_tree = 0
        self.root = IntervalBinaryTreeNode(0, 1, None, None, self.lip_const,
                                           self.glob_lower_bound, self.glob_upper_bound,
                                           self.glob_lower_bound, self.glob_upper_bound)
        self.nodes_expanded_at_each_height[0] = [(1, self.root)]
        self.expand_node(self.root)

    # Utilities ------------------------------------------------------------------------------------
    def _get_lower_upper_bounds_from_insert_idx(self, h, left_insert_idx):
        """ Returns lower and upper bound a node's child from it's index. """
        if left_insert_idx == 0:
            lower_bound_for_children = self.glob_lower_bound
        else:
            lower_bound_for_children = \
                self.nodes_expanded_at_each_height[h][left_insert_idx-1][1].Blb
        if left_insert_idx >= len(self.nodes_expanded_at_each_height[h]):
            upper_bound_for_children = self.glob_upper_bound
        else:
            upper_bound_for_children = \
                self.nodes_expanded_at_each_height[h][left_insert_idx][1].Bub
        return (lower_bound_for_children, upper_bound_for_children)

    def _get_left_insert_idx(self, h, k):
        """ Returns the index to insert the k'th node at height h into
            self.nodes_expanded_at_each_height.
        """
        return bisect.bisect([elem[0] for elem in self.nodes_expanded_at_each_height[h]], k)

    def get_lower_upper_bounds_for_unexpanded_node(self, h, k):
        """ Returns lower and upper bound a node (h, k) that has not been expanded. """
        if h > self.max_height_expanded:
            return (self.glob_lower_bound, self.glob_upper_bound)
        else:
            # First compute the lower upper bounds for the current h level.
            left_insert_idx = self._get_left_insert_idx(h, k)
            (lb_h, ub_h) = self._get_lower_upper_bounds_from_insert_idx(h, left_insert_idx)
            # Compute loewr and upper bounds for the height immediately below.
            ch_h, lch_k = get_left_child_h_k(h, k)
            (lb_ch, ub_ch) = self.get_lower_upper_bounds_for_unexpanded_node(ch_h, lch_k)
            return (max(lb_h, lb_ch), min(ub_h, ub_ch))

    def get_lower_upper_bounds_for_unexpanded_children(self, node):
        """ Returns lower and upper bound a node's children that has not been expanded. """
        ch_h, lch_k = get_left_child_h_k(node.h, node.k)
        return self.get_lower_upper_bounds_for_unexpanded_node(ch_h, lch_k)

    def expand_node(self, node):
        """ Expands the given node. """
        # Decide h, k for the children.
        ch_h, lch_k = get_left_child_h_k(node.h, node.k)
        rch_k = lch_k + 1
        # Create dictionary for the current height if not created already
        if not ch_h in self.nodes_expanded_at_each_height:
            self.nodes_expanded_at_each_height[ch_h] = []
        # Comute location for the children and the upper/lower bounds
        left_insert_idx = self._get_left_insert_idx(ch_h, lch_k)
        (lower_bound_for_children, upper_bound_for_children) = \
            self._get_lower_upper_bounds_from_insert_idx(ch_h, left_insert_idx)
        # Add left child
        node.set_left_child(
            IntervalBinaryTreeNode(ch_h, lch_k, node, True, self.lip_const,
                                   lower_bound_for_children, upper_bound_for_children,
                                   self.glob_lower_bound, self.glob_upper_bound))
        self.nodes_expanded_at_each_height[ch_h].insert(left_insert_idx,
                                                        (lch_k, node.left_child))
        # Add right child
        node.set_right_child(
            IntervalBinaryTreeNode(ch_h, rch_k, node, False, self.lip_const,
                                   lower_bound_for_children, upper_bound_for_children,
                                   self.glob_lower_bound, self.glob_upper_bound))
        self.nodes_expanded_at_each_height[ch_h].insert(left_insert_idx + 1,
                                                        (rch_k, node.right_child))
        # Finally set max_height_expanded
        if self.max_height_expanded < node.left_child.h:
            self.max_height_expanded = node.left_child.h

    def get_sub_interval_from_node(self, node):
        """ Returns the sub interval from the node. """
        return self.get_sub_interval_from_hk_vals(node.h, node.k)

    def get_sub_interval_from_hk_vals(self, h, k):
        """ Returns the sub interval from the height and k-index. """
        k = int(k)
        sub_int_width = self.int_width / (2**h)
        sub_int_lb = self.int_lb + sub_int_width * (k - 1)
        sub_int_ub = self.int_lb + sub_int_width * k
        return (sub_int_lb, sub_int_ub)

    # Methods for updating the upper and lower bounds on the estimates -----------------------------
    def _update_Blb_Bub_for_nodes_at_same_depth(self, node):
        """ If the Blb value of any node to the right is lower than this Blb value, it updates
            those nodes. Similarly, if the Bub value to the left is higher than this Bub value,
            it updates those nodes.
        """
        pivot_idx = bisect.bisect(
            [elem[0] for elem in self.nodes_expanded_at_each_height[node.h]], node.k)
        end_idx = len(self.nodes_expanded_at_each_height[node.h])
        for idx in range(pivot_idx, end_idx):
            if self.nodes_expanded_at_each_height[node.h][idx][1].Blb < node.Blb:
                self.nodes_expanded_at_each_height[node.h][idx][1].Blb = node.Blb
            else:
                break
        for idx in range(pivot_idx - 2, -1, -1):
            if self.nodes_expanded_at_each_height[node.h][idx][1].Bub > node.Bub:
                self.nodes_expanded_at_each_height[node.h][idx][1].Bub = node.Bub
            else:
                break

    def test_for_monotonicity(self):
        """ Tests for monotonicity. """
        for _, nodes_expanded_at_height_h in self.nodes_expanded_at_each_height.items():
            max_lb_at_height = self.glob_lower_bound
            max_ub_at_height = self.glob_lower_bound
            for _, node in nodes_expanded_at_height_h:
                # For current node
                assert max_lb_at_height <= node.Blb
                assert max_ub_at_height <= node.Bub
                max_lb_at_height = node.Blb
                max_ub_at_height = node.Bub

    def update_B_on_path(self, path):
        """ This updates the estimates on the path starting from the lowest node. """
        curr_node = path[-1]
        curr_idx = -1
        # update in case this is a leaf node
        if curr_node.is_a_leaf():
            (ch_lb, ch_ub) = self.get_lower_upper_bounds_for_unexpanded_children(curr_node)
            curr_node.Blb = max(curr_node.flb, curr_node.Blb, ch_lb)
            curr_node.Bub = min(curr_node.fub, curr_node.Bub, ch_ub)
            self._update_Blb_Bub_for_nodes_at_same_depth(curr_node)
            curr_node = curr_node.parent
            curr_idx -= 1
        while curr_node: # while curr_node is not None
            assert path[curr_idx] is curr_node
            # Below, we can avoid min(curr_node.left_child.Blb, curr_node.right_child.Blb) and
            # curr_node.right_child.Bub due to monomotonicty.
            curr_node.Blb = max(curr_node.flb, curr_node.Blb, curr_node.left_child.Blb)
            curr_node.Bub = min(curr_node.fub, curr_node.Bub, curr_node.right_child.Bub)
            self._update_Blb_Bub_for_nodes_at_same_depth(curr_node)
            curr_idx -= 1
            curr_node = curr_node.parent

    def _set_node_flb_fub_vals(self, node):
        """ Sets the flb and fub values for the node. """
        node.sum_of_SG_consts = np.array(node.data_inv_sigmasq_vals_for_node).sum()
        if node.sum_of_SG_consts <= 0:
            node.flb = self.glob_lower_bound
            node.fub = self.glob_upper_bound
            return
        fbar = (1/node.sum_of_SG_consts) * \
               (np.array(node.data_y_vals_for_node) *
                np.array(node.data_inv_sigmasq_vals_for_node)).sum()
        uncert = self.lip_const * self.int_width / (2**node.h) + \
                 self.beta_t(self.num_data_in_tree) / np.sqrt(node.sum_of_SG_consts)
        node.flb = fbar - uncert
        node.fub = fbar + uncert

    def update_B_on_entire_tree(self):
        """ This updates the estimates on the entire tree. """
        # pylint: disable=too-many-branches
        # First update the fub, flb values ---------------------------------------------------------
        for _, nodes_at_h in self.nodes_expanded_at_each_height.items():
            for _, node in nodes_at_h:
                self._set_node_flb_fub_vals(node)
        # Update Bub/Blb values while correcting for monotonicity ----------------------------------
        height_vals = sorted(list(self.nodes_expanded_at_each_height))
        for h in reversed(height_vals):
            nodes_at_h = self.nodes_expanded_at_each_height[h]
            # Lower bound should be non-decreasing from the left
            lb_max_at_height_k = self.glob_lower_bound
            for _, node in nodes_at_h:
                if node.is_a_leaf():
                    (lower_bound_for_children, upper_bound_for_children) = \
                        self.get_lower_upper_bounds_for_unexpanded_children(node)
                    # Store this temporarily so that we can re-use it. Will delete later.
                    node.upper_bound_for_children = upper_bound_for_children
                    pre_Blb = max(node.flb, node.Blb, lower_bound_for_children)
                else:
                    # Below, we can avoid min(curr_node.left_child.Blb, curr_node.right_child.Blb)
                    # and curr_node.right_child.Bub due to monomotonicty.
                    pre_Blb = max(node.flb, node.Blb, node.left_child.Blb)
                if pre_Blb > lb_max_at_height_k:
                    lb_max_at_height_k = pre_Blb
                    node.Blb = pre_Blb
                else:
                    node.Blb = lb_max_at_height_k
            # Upper bound should be non-increasing from the right
            ub_min_at_height_k = self.glob_upper_bound
            for _, node in reversed(nodes_at_h):
                if node.is_a_leaf():
                    pre_Bub = min(node.fub, node.Bub, node.upper_bound_for_children)
                    delattr(node, 'upper_bound_for_children')
                else:
                    pre_Bub = min(node.fub, node.Bub, node.right_child.Bub)
                if pre_Bub < ub_min_at_height_k:
                    ub_min_at_height_k = pre_Bub
                    node.Bub = pre_Bub
                else:
                    node.Bub = ub_min_at_height_k

    # Methods for obtaining recommendations and adding data points ---------------------------------
    def get_recommendation(self, perf_goal):
        """ Obtain a recommendation. """
        # pylint: disable=arguments-differ
        opt_path = self.opt_traverse(perf_goal)
        opt_node = opt_path[-1]
        opt_interval = self.get_sub_interval_from_node(opt_node)
        rec = opt_interval[0] + np.random.random() * (opt_interval[1] - opt_interval[0])
        logger.debug('Returning reco %0.3f for %s with %d data (h=%d).',
                     rec, self.name, self.num_data_in_tree, self.max_height_expanded)
        return rec

    def opt_traverse(self, perf_goal):
        """ Traverses thre self to find the optimum. """
        curr_node = self.root
        path = [curr_node]
        while (curr_node is not None) and \
              (curr_node.sum_of_SG_consts >= self.tau_ht(curr_node.h, self.num_data_in_tree)) and \
              (curr_node.left_child is not None):
            if (curr_node.left_child.get_B_val(perf_goal) >=
                curr_node.right_child.get_B_val(perf_goal)):
                curr_node = curr_node.left_child
            else:
                curr_node = curr_node.right_child
            path.append(curr_node)
        return path

    def get_recommendation_for_upper_bound(self, perf_goal):
        """ Obtain a recommendation for estimating the upper bound. """
        # pylint: disable=arguments-differ
        ub_path = self.ub_traverse(perf_goal)
        ub_node = ub_path[-1]
        ub_interval = self.get_sub_interval_from_node(ub_node)
        rec = ub_interval[1]
        return rec

    def ub_traverse(self, perf_goal):
        """ Traverses the tree to find an upper bound. """
        curr_node = self.root
        path = [curr_node]
        while (curr_node is not None) and \
              (curr_node.sum_of_SG_consts >= self.tau_ht(curr_node.h, self.num_data_in_tree)) and \
              (curr_node.left_child is not None):
            if curr_node.right_child.Blb >= perf_goal:
                curr_node = curr_node.left_child
            else:
                curr_node = curr_node.right_child
            path.append(curr_node)
        return path

    def get_recommendation_for_lower_bound(self, perf_goal):
        """ Obtain a recommendation for estimating the upper bound. """
        # pylint: disable=arguments-differ
        lb_path = self.lb_traverse(perf_goal)
        lb_node = lb_path[-1]
        lb_interval = self.get_sub_interval_from_node(lb_node)
        rec = lb_interval[0]
        return rec

    def lb_traverse(self, perf_goal):
        """ Traverses the tree to find an upper bound. """
        curr_node = self.root
        path = [curr_node]
        while (curr_node is not None) and \
              (curr_node.sum_of_SG_consts >= self.tau_ht(curr_node.h, self.num_data_in_tree)) and \
              (curr_node.left_child is not None):
            if curr_node.left_child.Bub <= perf_goal:
                curr_node = curr_node.right_child
            else:
                curr_node = curr_node.left_child
            path.append(curr_node)
        return path

    def _assign_point_to_node(self, node, x, y, sigma):
        """ Assigns the value to the node and updates the values. """
        node.add_data_point(y_val=y, sigma=sigma, x_val=x)
        self._set_node_flb_fub_vals(node)

    def assign_point_to_nodes_on_path(self, x, y, sigma):
        """ Assigns the point (x, y) to nodes on a path P and returns P. """
        # First select the path and update the flb/fub values on the path.
        curr_node = self.root
        self._assign_point_to_node(curr_node, x, y, sigma)
        path = [curr_node]
        while (not curr_node.is_a_leaf()) and \
              (curr_node.sum_of_SG_consts >= self.tau_ht(curr_node.h, self.num_data_in_tree)):
            lhk, uhk = self.get_sub_interval_from_node(curr_node)
            mid_point = 0.5 * (lhk + uhk)
            if x < mid_point:
                curr_node = curr_node.left_child
            else:
                curr_node = curr_node.right_child
            self._assign_point_to_node(curr_node, x, y, sigma)
            path.append(curr_node)
        # Next update the Blb/Bub values on the path
        self.update_B_on_path(path)
        # Finally, decide whether to expand the last child or not
        last_node = path[-1]
        if (last_node.is_a_leaf()) and \
           (last_node.num_data_in_node >= MIN_NUM_DATA_IN_NODE_BEFORE_EXPANDING) and \
           (last_node.sum_of_SG_consts >= self.tau_ht(last_node.h, self.num_data_in_tree)):
            self.expand_node(last_node)
#         print('(h,k) = (%d,%d):: SG-consts=%0.5f, tau_ht=%0.4f, num_points=%d'%(
#             last_node.h, last_node.k, last_node.sum_of_SG_consts,
#             tau_ht(last_node.h, self.num_data_in_tree),
#             last_node.num_data_in_node))
        return path

    def update_model_with_new_data(self, Allocs, Rewards, Loads, Sigmas, _):
        """ Updates model with new data. """
        # Event end times not necessary here.
        X = [alloc/load for alloc, load in zip(Allocs, Loads)]
        for x, y, sigma in zip(X, Rewards, Sigmas):
            self.add_data_point(x, y, sigma)

    def add_data_point(self, x, y, sigma):
        """ Adds a single data point. """
        if (x <= 0) or (sigma <= 0) or (not np.isfinite(sigma)):
#             logger.info('Skipping point (%0.3f, %0.3f, %0.3f)', x, y, sigma)
            return
#         if x > self.int_ub or x < self.int_lb:
#             print('Received point (x, y, sigma)', (x, y, sigma))
        self.all_data.append((x, y, sigma))
        self.num_data_in_tree += 1
        self.assign_point_to_nodes_on_path(x, y, sigma)
        # Update B on path
        if abs(self.num_data_in_tree - 2**(np.ceil(np.log2(self.num_data_in_tree)))) < 1e-5:
            self.update_B_on_entire_tree()

    def add_multiple_data_points(self, X, Y, Sigmas):
        """ Adds multiple data points. """
        lenX = len(X)
        if len(Y) != lenX or len(Sigmas) != lenX:
            raise ValueError('Lenths of X, Y, Sigmas should be equal. Received ' +
                             'len(X)=%d, len(Y)=%d, len(Sigmas)=%d.'%(lenX, len(Y), len(Sigmas)))
        for (x, y, sigma) in zip(X, Y, Sigmas):
            self.add_data_point(x, y, sigma)

    # Utilities used for inference -----------------------------------------------------------------
    def compute_conf_interval_for_input(self, x):
        """ Obtains a confidence interval for the given point. """
        # First find a path from root to leaf which contains x
        curr_node = self.root
        parent_node = None
        lcb = self.glob_lower_bound
        ucb = self.glob_upper_bound
        while curr_node: # curr_node is not None
            lcb = max(lcb, curr_node.Blb)
            ucb = min(ucb, curr_node.Bub)
            # Move down the path
            parent_node = curr_node
            lhk, uhk = self.get_sub_interval_from_node(parent_node)
            mid_point = 0.5 * (lhk + uhk)
            if x < mid_point:
                curr_node = parent_node.left_child
            else:
                curr_node = parent_node.right_child
        # Finally, obtain lower/upper bounds from un-expanded children.
        lower_bound_for_children, upper_bound_for_children = \
            self.get_lower_upper_bounds_for_unexpanded_children(parent_node)
        lcb = max(lcb, lower_bound_for_children)
        ucb = min(ucb, upper_bound_for_children)
        return (lcb, ucb)

    def compute_estimate_for_input(self, x):
        """ Obtains an estimate (along with ucbs and lcbs) for the given point. """
        lcb, ucb = self.compute_conf_interval_for_input(x)
        est = (lcb + ucb) / 2
        return est, lcb, ucb

