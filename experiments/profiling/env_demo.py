"""
    Generate the environment for the demo.
    -- kirthevasank
"""

# pylint: disable=import-error

# Local
from cilantro.core.henv import InternalNode, LinearLeafNode, TreeEnvironment
# In Demo
from propalloc_workload_generator import PropAllocWorkloadGenerator
from workloads.k8s_proportional_data_source import get_abs_unit_demand


def generate_env(cluster_type):
    """ Generate a synthetic organisational tree. """
    # Create the environment -----------------------------------------------------------------------
    root = InternalNode('root')
    child1 = InternalNode('c1')
    child2 = LinearLeafNode('c2', threshold=2.1)
    root.add_children([child1, child2], [1, 1])
    child11 = LinearLeafNode('c11', threshold=0.6)
    child12 = LinearLeafNode('c12', threshold=7.2)
    child1.add_children([child11, child12], [2, 1])
    env = TreeEnvironment(root, 1)

    # Create the workload --------------------------------------------------------------------------
    propworkgen = PropAllocWorkloadGenerator(cluster_type=cluster_type)
    def generate_propalloc_workload_info(leaf: LinearLeafNode, workload_type):
        [_, weight, _] = leaf.parent.children[leaf.name]
        path = leaf.get_path_from_root()
        workload_server_objs = propworkgen.generate_workload_server_objects(
            app_name=path, threshold=leaf.threshold, app_weight=weight,
            app_unit_demand=get_abs_unit_demand(leaf.threshold))
        workload_cilantro_client_objs = propworkgen.generate_cilantro_client_objects(app_name=path)
        k8s_objects = [*workload_server_objs, *workload_cilantro_client_objs]
        leaf.update_workload_info({"k8s_objects": k8s_objects,
                                   "workload_type": workload_type})
    # Add workload information to clients ----------------------------------------------------------
    generate_propalloc_workload_info(child2, 'dummy1')
    generate_propalloc_workload_info(child11, 'dummy2')
    generate_propalloc_workload_info(child12, 'dummy1')
    return env

