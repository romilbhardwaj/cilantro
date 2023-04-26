"""
    A hierarchical environment for fair sharing.
    -- romilbhardwaj
    -- kirthevasank
"""


import logging
import numpy as np


logger = logging.getLogger(__name__)


TREE_PATH_DELIMITER = '--'

class TreeEnvNode:
    """ An abstract class for nodes in the Tree Environment. """

    def __init__(self, name):
        """ Constructor. """
        if TREE_PATH_DELIMITER in name:
            raise ValueError('Node name (%s) cannot contain character %s'%(
                name, TREE_PATH_DELIMITER))
        self.name = name
        self.parent = None
        self._curr_demand = None
        self._capacity = np.inf
        self._path_from_root = None

    def reset_demand(self):
        """ Resets current demand. """
        self._curr_demand = None

    def set_capacity(self, capacity):
        """ Sets capacity. """
        self._capacity = capacity

    def generate_curr_capacity(self):
        """ Generate the current capacity. """
        self._capacity = self._child_generate_curr_capacity()

    def get_curr_capacity(self):
        """ Returns current capacity. """
        return self._capacity

    def reset_capacity(self):
        """ Reset capacity. """

    @classmethod
    def _child_generate_curr_capacity(cls):
        """ Set current capacity. Can be over-ridden by a child class. """
        return np.inf

    def compute_curr_demand(self):
        """ Compute current demand. """
        self._curr_demand = min(self._child_compute_curr_demand(), self._capacity)
        return self._curr_demand

    def _child_compute_curr_demand(self):
        """ Compute current demand. """
        raise NotImplementedError('Implement in a child class.')

    def get_curr_demand(self):
        """ Get current demand. """
        return self._curr_demand

    def set_parent(self, parent):
        """ Sets parent. """
        self.parent = parent

    def set_path_from_root(self, path_from_root):
        """ Sets path from root. """
        self._path_from_root = path_from_root

    def get_path_from_root(self):
        """ Return path from root. """
        if self._path_from_root:
            return self._path_from_root
        else:
            raise ValueError('path_from_root not set for node %s with parent %s.'%(
                self.name, self.parent))

    def __str__(self):
        """ Return string. """
        if self._path_from_root is None:
            return self.name
        else:
            return self._path_from_root


class LeafNode(TreeEnvNode):
    """ A class for leaf nodes, which usually is an application node. """

    def __init__(self, name, threshold=None, util_scaling='linear',
                 update_load_on_each_serve=False):
        """ Construtor. """
        super().__init__(name)
        self.threshold = threshold
        self._curr_load = None
        self.util_scaling = util_scaling
        self.update_load_on_each_serve = update_load_on_each_serve
        self.workload_info = {}

    def get_norm_util_from_reward(self, reward):
        """ Computes the normalised utility from the reward. """
        norm_val = min(1.0, reward/self.threshold)
        if self.util_scaling == 'linear':
            return norm_val
        elif self.util_scaling == 'quadratic':
            return norm_val ** 2
        elif self.util_scaling == 'sqrt':
            return np.sqrt(norm_val)
        else:
            return self.util_scaling(norm_val)

    def update_workload_info(self, info):
        """ Updates the workload info. """
        for key, val in info.items():
            self.workload_info[key] = val

    def get_workload_info(self, key: str):
        """ Gets a value for the corresponding in workload info. """
        return self.workload_info[key]

    def set_path_from_root(self, path_from_root):
        """ Sets path from root. """
        super().set_path_from_root(path_from_root)
        self.workload_info['path_from_root'] = self._path_from_root

    def delete_keys_from_workload_info(self, keys_to_delete):
        """ Delete some keys from workload_info. """
        for key in keys_to_delete:
            self.workload_info.pop(key)

    def get_curr_load(self):
        """ Return the current load. """
        return self._curr_load

    def set_curr_load(self, load):
        """ Return the current load. """
        # Use set_curr_load if you wish to manually set the load for a user in a time step.
        # Use generate_curr_load if you wish to use a method to generate the loads.
        self._curr_load = load

    def reset_load(self):
        """ Resets load. """
        self._curr_load = None

    def generate_curr_load(self):
        """ Generate the current load using some generator. """
        # Use set_curr_load if you wish to manually set the load for a user in a time step.
        # Use generate_curr_load if you wish to write a method to generate the loads.
        self._curr_load = self._child_generate_curr_load()
        return self._curr_load

    def _child_generate_curr_load(self):
        """ Return the current load. """
        raise NotImplementedError('Implement in a child class.')

    def _child_compute_curr_demand(self):
        """ Compute the demand for the current load. """
        return self._child_compute_demand_for_load(self._curr_load)

    def compute_demand_for_load(self, load):
        """ Compute the demand for the current load. """
        return self._child_compute_demand_for_load(load)

    def _child_compute_demand_for_load(self, load):
        """ Return the demand for the current load. """
        raise NotImplementedError('Implement in a Child class.')

    def serve_curr_load(self, alloc):
        """ Serve the current load. """
        ret = self._child_serve_curr_load(alloc)
        if self.update_load_on_each_serve:
            self._curr_load = None
        return ret

    def _child_serve_curr_load(self, alloc):
        """ Serve the current load. """
        raise NotImplementedError('Implement in a child class.')


class LinearLeafNode(LeafNode):
    """ A class for agents whose demand scales linearly with load. """

    def __init__(self, name, threshold=None, util_scaling='linear', unit_demand=None):
        """ Constructor. """
        super().__init__(name, threshold)
        self.unit_demand = unit_demand

    def set_unit_demand(self, unit_demand):
        """ Sets unit demand. """
        self.unit_demand = unit_demand

    def get_payoff(self, alloc, load):
        """ Returns payoff for allocation. """
        return self._get_payoff_per_unit_load(alloc/load)

    def _get_payoff_per_unit_load(self, alloc_per_unit_load):
        """ Returns payoff for unit allocation. """
        raise NotImplementedError('Implement in a child class.')

    def _child_compute_demand_for_load(self, load):
        """ Return the demand for the load. """
        if self.unit_demand is None:
            return None
        elif load in [None, np.inf]:
            return np.inf
        else:
            return self.unit_demand * load


class InternalNode(TreeEnvNode):
    """ A class for internal nodes. """

    def __init__(self, name, children=None, weights=None):
        """ An internal node in the tree hierarchy. """
        # Declare attributes
        super().__init__(name)
        # Children will be a dictionary where the key is each node's name and the value is a
        # list of 3 items in the order [node, weight, entitlement].
        self.children = {}
        self.num_children = None
        # Add children -----------------------------------------------------------------------
        if children is None:
            children = []
            weights = []
        else:
            weights = weights if weights else [1] * len(self.children)
        self.add_children(children, weights)

    def add_children(self, nodes, weights):
        """ Adds a child node. """
        if not hasattr(nodes, '__iter__'):
            nodes = [nodes]
            weights = [weights]
        assert len(nodes) == len(weights)
        for (node, weight) in zip(nodes, weights):
            if node.name in self.children.keys():
                raise ValueError('%s appears twice in children of %s'%(node.name, self.name))
            self.children[node.name] = [node, weight, None] # Will compute entitlement shortly.
            node.set_parent(self)
        self.num_children = len(self.children)
        self._compute_local_entitlements()

    def remove_child(self, child_name, ignore_exception=False):
        """ Removes child. """
        try:
            self.children.pop(child_name)
        except KeyError as key_err:
            if not ignore_exception:
                raise ValueError('%s child not found in node in %s.'%(child_name, self.name)) \
                    from key_err

    def is_a_child(self, node):
        """ Returns true if child is a child. """
        return node in [elem[0] for elem in self.children]

    def _compute_local_entitlements(self):
        """ Computes entitlements for the current siblings. """
        if self.num_children > 0:
            weights = [val[1] for _, val in self.children.items()]
            weight_sum = sum(weights)
            for _, val in self.children.items():
                val[2] = val[1]/weight_sum

    def _child_compute_curr_demand(self):
        """ Computes the current demand. """
        curr_demand = 0
        for _, child_node_info in self.children.items():
            curr_demand += child_node_info[0].compute_curr_demand()
        return curr_demand

    def reset_demand(self):
        """ Resets demand for this node and its children. """
        self._curr_demand = None
        for _, child_node in self.children.items():
            child_node.reset_demand()


class TreeEnvironment:
    """ An environment with a hierarchical tree structure. """

    def __init__(self, root, num_resource_types):
        """ Constructor. """
        self.root = root
        self.num_resource_types = num_resource_types
        self._str_descr = None
        self._entitlements = None
        self.leaf_nodes = {} # leaf_nodes is a dictionary of leaf nodes of the form path:node
                             # where path is a string specifying the path to the node and node is a
                             # LeafNode object.
        self.all_nodes = {} # Similar structure to leaf_nodes, but for all nodes.
        self.update_tree()

    def update_tree(self):
        """ Updates the tree starting at the root. """
        if self.root:
            self._update_subtree('', self.root)

    def __str__(self):
        """ Returns description of the environment. """
        if not self._str_descr:
            _, leaf_list = self.get_entitlements(ret_str=True)
            descr = 'Env(#nodes=%d, #leaf-nodes=%d)'%(len(self.all_nodes), len(self.leaf_nodes))
            self._str_descr = descr + ':: ' + ', '.join(leaf_list) + '.'
        return self._str_descr

    def get_num_leaf_nodes(self):
        """ Returns the number of leaf nodes. """
        return len(self.leaf_nodes)

    def get_entitlements(self, ret_str=False):
        """ Returns entitlements. """
        # pylint: disable=import-outside-toplevel
        if not self._entitlements or not self._entitlements[0]:
            from cilantro.policies.prop_fairness import PropFairness
            self_copy = self
            prop_policy = PropFairness(self_copy, resource_quantity=1.0)
            prop_policy.initialise()
            inf_loads = {leaf_path:np.inf for leaf_path in self_copy.leaf_nodes}
            entitlements = prop_policy.get_resource_allocation(inf_loads)
            self.reset_loads()
            leaf_list = ['(%s, e%0.2f, t%0.2f)'%(leaf.name, entitlements[leaf_path], leaf.threshold)
                         for leaf_path, leaf in self_copy.leaf_nodes.items()]
            self._entitlements = (entitlements, leaf_list)
        if ret_str:
            return self._entitlements
        else:
            return self._entitlements[0]

    def _check_path_name_compliance(self, path_name):
        """ Check path name compliance. """
        if path_name in self.all_nodes.keys():
            raise ValueError('Path name %s already exists in the tree.'%(path_name))

    @classmethod
    def _path_join(cls, curr_path, node_name):
        """ Join path. """
        if curr_path:
            return curr_path + TREE_PATH_DELIMITER + node_name
        else:
            return node_name

    def _update_subtree(self, start_path, start_node, prefix=None):
        """ Update the leaf nodes. """
        prefix = prefix if prefix else []
        curr_path = self._path_join(start_path, start_node.name)
        self.all_nodes[curr_path] = start_node
        start_node.set_path_from_root(curr_path)
        if isinstance(start_node, LeafNode):
            self.leaf_nodes[curr_path] = start_node
        else:
            for _, child_info in start_node.children.items():
                node = child_info[0]
                self._update_subtree(curr_path, node)

    def get_curr_loads(self):
        """ Return loads for each leaf node. """
        return {leaf_path: leaf.get_curr_load() for leaf_path, leaf in self.leaf_nodes.items()}

    def set_capacities_for_nodes(self, capacities):
        """ Sets capacities for all nodes. """
        for node_path, cap in capacities.items():
            assert node_path in self.all_nodes
            self.all_nodes[node_path].set_capacity(cap)

    def generate_loads_for_all_leaf_nodes(self):
        """ Generates loads for all leaf nodes. """
        # pylint: disable=protected-access
        for _, leaf in self.leaf_nodes.items():
            leaf.generate_curr_load()

    def generate_capacities_for_all_nodes(self):
        """ Generates capacities for all nodes. """
        # pylint: disable=protected-access
        for _, node in self.all_nodes.items():
            node.generate_curr_capacity()

    def get_curr_capacities_for_all_nodes(self):
        """ Gets capacities for all nodes. """
        return {node_path:node.get_curr_capacity()
                for node_path, node in self.all_nodes.items()}

    def get_curr_capacities_for_all_leaf_nodes(self):
        """ Gets capacities for all nodes. """
        return {leaf_path:leaf.get_curr_capacity()
                for leaf_path, leaf in self.leaf_nodes.items()}

    def reset_loads(self):
        """ Resets loads for each leaf node. """
        for _, leaf in self.leaf_nodes.items():
            leaf.reset_load()

    def reset_capacities(self):
        """ Resets capacities. """
        for _, node in self.all_nodes.items():
            node.reset_capacity()

    def get_curr_demand(self):
        """ Return demands for each leaf node. """
        curr_loads = self.get_curr_loads()
        return self.get_demands_for_loads(curr_loads)

    def get_demands_for_loads(self, curr_loads):
        """ Return demands for the load. """
        return {leaf_path: self.leaf_nodes[leaf_path].compute_demand_for_load(load)
                for leaf_path, load in curr_loads.items()}

    def allocate_and_get_feedback(self, allocs, loads_to_serve):
        """ Allocate the allocations to each leaf node and get feedback.
            feedback_type shoule be either 'reward' or 'payoff'.
        """
        return {leaf_path: leaf.serve_curr_load(allocs[leaf_path], loads_to_serve[leaf_path]) for
                leaf_path, leaf in self.leaf_nodes}

    def delete_subtree(self, sub_tree_root):
        """ Delete the sub tree starting from sub_tree_root. """
        nodes_to_delete = []
        leaf_nodes_to_delete = []
        def _dfs_add_to_del_list(node):
            """ Depth first search. """
            nodes_to_delete.append(node.get_path_from_root())
            if isinstance(node, LeafNode):
                leaf_nodes_to_delete.append(node.get_path_from_root())
            else:
                for _, child_info in node.children:
                    _dfs_add_to_del_list(child_info[0])
        # First make a list of nodes to delete
        _dfs_add_to_del_list(sub_tree_root)
        # Delete this child from the parent
        parent_node = sub_tree_root.parent
        parent_node.remove_child(sub_tree_root.name)
        # Delete all children from list
        for path_name in nodes_to_delete:
            self.all_nodes.pop(path_name)
        for path_name in leaf_nodes_to_delete:
            self.leaf_nodes.pop(path_name)

    def add_nodes_to_tree_from_path(self, path, weights=None, leaf_threshold=None,
                                    leaf_unit_demand=None, leaf_util_scaling=None,
                                    last_node_is_a_leaf_node=True,
                                    update_tree_at_end=True):
        """ Given a path, adds nodes to the tree. """
        # pylint: disable=too-many-branches
        if path in self.leaf_nodes:
            return False
        nodes_in_path = path.split(TREE_PATH_DELIMITER)
        num_nodes_in_path = len(nodes_in_path)
        if not weights:
            # Use default weight 1
            weights = [1] * num_nodes_in_path
        elif not isinstance(weights, list):
            # Set weights for all to be the same
            weights = [weights] * num_nodes_in_path
        # Add leaves to the tree one by one --------------------------------------------------------
        curr_node = None
        new_node_added = False
        for idx, node_name in enumerate(nodes_in_path):
            if idx == 0:
                if self.root is None:
                    self.root = InternalNode(node_name)
                    new_node_added = True
                curr_node = self.root
            else:
                if not (node_name in curr_node.children):
                    if idx + 1 == num_nodes_in_path and last_node_is_a_leaf_node:
                        new_node = LinearLeafNode(name=node_name, threshold=leaf_threshold,
                                                  util_scaling=leaf_util_scaling,
                                                  unit_demand=leaf_unit_demand)
                    else:
                        new_node = InternalNode(name=node_name)
                    new_node_added = True
                    curr_node.add_children([new_node], [weights[idx]])
                else:
                    if idx + 1 == num_nodes_in_path:
                        print('Requested to add %s to tree, but node already present.'%(path))
                curr_node = curr_node.children[node_name][0]
        if not new_node_added:
            raise KeyError(f"Path {path} was not added. Are you sure it's the correct path format?")
        self._entitlements = None # This needs to be recomputed
        if update_tree_at_end:
            self.update_tree()
        return True
#         if new_node_added:
#             self._entitlements = None # This needs to be recomputed
#         if new_node_added and update_tree_at_end:
#             self.update_tree()
#         return new_node_added

    def _get_nodes_and_weights_in_bfs_order(self):
        """ get nodes and weights in BFS order. """
        ret = [(self.root, self.root.name, -1)]
        curr_idx = 0
        while curr_idx < len(ret):
            curr_node, curr_path, _ = ret[curr_idx]
            if isinstance(curr_node, InternalNode):
                node_path_prefix = curr_path + TREE_PATH_DELIMITER
                nodes_to_add = [(val[0], node_path_prefix + val[0].name, val[1])
                                for _, val in curr_node.children.items()]
                ret.extend(nodes_to_add)
            curr_idx += 1
        return ret

    def get_leaf_node_paths(self):
        """ Returns the paths of all leaf nodes. """
        return list(self.leaf_nodes.keys())

    def write_to_file(self, file_name):
        """ Writes tree to a file. """
        nodes_and_weights_in_bfs_order = self._get_nodes_and_weights_in_bfs_order()
        internal_str_list = []
        leaf_str_list = []
        for node, path, weight in nodes_and_weights_in_bfs_order:
            if isinstance(node, InternalNode):
                internal_str_list.append('in ' + path + ' ' + str(weight))
            else:
                threshold_str = str(node.threshold) if isinstance(node.threshold, (int, float)) \
                                else '-1'
                ud_str = str(node.unit_demand) if isinstance(node.unit_demand, (int, float)) \
                         else '-1'
                util_scaling_str = node.util_scaling if isinstance(node.util_scaling, str) \
                                   else '-1'
                leaf_str_list.append('lf ' + path + ' ' + str(weight) + ' ' + threshold_str +
                                     ' ' + ud_str + ' ' + util_scaling_str)
        ret = '\n'.join(internal_str_list + leaf_str_list)
        if file_name:
            with open(file_name, 'w') as write_file:
                write_file.write(ret)
        return ret


def load_env_from_file(file_name):
    """ Loads an environment from file. """
    with open(file_name, 'r') as read_file:
        lines = read_file.read().splitlines()
    # First create an an environment object --------------------------------------------------
    env = TreeEnvironment(None, 1)
    for line in lines:
        elems = line.split(' ')
        path = elems[1]
        path_length = len(path.split(TREE_PATH_DELIMITER))
        weights = [None] * (path_length - 1) + [float(elems[2])]
        if elems[0] == 'lf':
            leaf_threshold = None if elems[3] == '-1' else float(elems[3])
            leaf_unit_demand = None if elems[4] == '-1' else float(elems[4])
            if len(elems) >= 6:
                leaf_util_scaling = elems[5]
            else:
                leaf_util_scaling = 'linear'
            if leaf_util_scaling == '-1':
                logger.debug('Unrecognized unit scaling %s. Using linear scaling',
                             leaf_util_scaling)
                leaf_util_scaling = 'linear'
            last_node_is_a_leaf_node = True
        else:
            leaf_threshold = None
            leaf_unit_demand = None
            leaf_util_scaling = None
            last_node_is_a_leaf_node = False
        env.add_nodes_to_tree_from_path(path, weights, leaf_threshold, leaf_unit_demand,
                                        leaf_util_scaling, last_node_is_a_leaf_node,
                                        update_tree_at_end=False)
    env.update_tree()
    return env


def are_two_nodes_equal(node1, node2):
    """ Returns true if two nodes are equal. """
    if node1.name != node2.name:
        return False
    if isinstance(node1, InternalNode) and isinstance(node2, LeafNode):
        return False
    elif isinstance(node2, InternalNode) and isinstance(node1, LeafNode):
        return False
    elif isinstance(node1, InternalNode):
        return are_two_internal_nodes_equal(node1, node2)
    elif isinstance(node2, LeafNode):
        return are_two_leaf_nodes_equal(node1, node2)
    else:
        raise ValueError('Unknown types for node1 (%s) and node2 (%s).'%(
            str(type(node1)), str(type(node2))))

def are_two_internal_nodes_equal(node1, node2):
    """ Returns True if two internal nodes are equal. """
    if node1.num_children != node2.num_children:
        return False
    weights1 = np.array([val[1] for _, val in node1.children.items()])
    weights2 = np.array([val[1] for _, val in node2.children.items()])
    if np.linalg.norm(weights1 - weights2) > 1e-5:
        return False
    return True

def are_two_leaf_nodes_equal(leaf1, leaf2):
    """ Returns True if two leaf nodes are equal. """
    if abs(leaf1.threshold - leaf2.threshold) > 1e-5:
        return False
    return True

def are_two_environments_equal(env1, env2):
    """ Returns true if two environments are equal. """
    if not (env1.get_entitlements() == env2.get_entitlements()):
        return False
    # Cross-check each leaf in env1 with env2 --------------------------------------------------
    if not (len(env1.all_nodes) == len(env2.all_nodes)):
        return False
    for node_path, node1 in env1.all_nodes.items():
        node2 = env2.all_nodes[node_path]
        if not are_two_nodes_equal(node1, node2):
            return False
    return True

