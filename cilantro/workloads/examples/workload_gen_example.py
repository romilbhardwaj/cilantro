import logging

from cilantro.core.henv import InternalNode, LinearLeafNode, TreeEnvironment
from cilantro.workloads.k8s_workload_deployer import K8sWorkloadDeployer
from cilantro.workloads.propalloc import PropAllocWorkloadGenerator

logging.basicConfig(level=logging.DEBUG)


def generate_env():
    """ Generate a synthetic organisational tree. """
    propworkgen = PropAllocWorkloadGenerator()

    def generate_propalloc_workload_info(leaf: LinearLeafNode):
        [_, weight, _] = leaf.parent.children[leaf.name]
        path = leaf.get_path_from_root()
        workload_server_objs = propworkgen.generate_workload_server_objects(app_name=path,
                                                                            threshold=leaf.threshold,
                                                                            app_weight=weight,
                                                                            app_unit_demand=leaf.unit_demand)
        workload_cilantro_client_objs = propworkgen.generate_cilantro_client_objects(app_name=path)
        k8s_objects = [*workload_server_objs, *workload_cilantro_client_objs]
        leaf.update_workload_info({"k8s_objects": k8s_objects})

    root = InternalNode('root')
    child1 = InternalNode('c1')
    child2 = LinearLeafNode('c3', threshold=10)
    root.add_children([child1, child2], [1, 1])
    child11 = LinearLeafNode('c11', threshold=5)
    child12 = LinearLeafNode('c12', threshold=42)
    child1.add_children([child11, child12], [1, 1])
    env = TreeEnvironment(root, 1)

    # Add workload information to clients ----------------------------------------------------------
    generate_propalloc_workload_info(child2)
    generate_propalloc_workload_info(child11)
    generate_propalloc_workload_info(child12)
    return env


env = generate_env()
workload_exec = K8sWorkloadDeployer()

# Parse environment and create the workloads in kubernetes.
# This is equivalent to generating a YAML for each workload
# and running kubectl apply -f <yaml> for each workload.

workload_exec.deploy_environment(env)


# DEBUG: Convert to JSON and dump objects:

from kubernetes import client, config
import json
svc = workload_exec.parse_env_to_workloads(env)[0]
with open('output.json', "w") as f:
    json.dump(client.ApiClient().sanitize_for_serialization(svc), f, indent=4)