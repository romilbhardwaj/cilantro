import logging

from cilantro.core.henv import InternalNode, LinearLeafNode, TreeEnvironment
from cilantro.workloads.cassandrahorz_workload_generator import CassandraHorzWorkloadGenerator
from cilantro.workloads.k8s_workload_deployer import K8sWorkloadDeployer
from cilantro.workloads.cassandra_workload_generator import CassandraWorkloadGenerator

logging.basicConfig(level=logging.DEBUG)


def generate_env():
    """ Generate a synthetic organisational tree. """
    workgen = CassandraHorzWorkloadGenerator(cilantro_image="public.ecr.aws/cilantro/cilantro:latest")

    def generate_workload_info(leaf: LinearLeafNode):
        [_, weight, _] = leaf.parent.children[leaf.name]
        path = leaf.get_path_from_root()
        workload_info = leaf.workload_info
        workload_server_objs = workgen.generate_workload_server_objects(app_name=path,
                                                                        threshold=leaf.threshold,
                                                                        app_weight=weight,
                                                                        app_unit_demand=leaf.unit_demand,
                                                                        **workload_info)

        # Generate workload client. For the cassandra workload, the cilantro client is
        # embedded with the workload client. In other words, the cilantro client is a
        # part of workload_client_objs here.

        workload_client_objs = workgen.generate_workload_client_objects(app_name=path,
                                                                        threshold=leaf.threshold,
                                                                        **workload_info)

        # generate_cilantro_client_objects returns empty list for Cassandra workload
        # because the cilantro client is embedded with the workload client.

        # It is still a good practice to call generate_cilantro_client_objects and append it
        # to the k8s_objects list because other workloads may have separate cilantro clients.
        cilantro_client_objs = workgen.generate_cilantro_client_objects(app_name=path)

        # Add to k8s_objects.
        k8s_objects = [*workload_server_objs, *workload_client_objs, *cilantro_client_objs]
        leaf.update_workload_info({"k8s_objects": k8s_objects})

    root = InternalNode('root')
    # child1 = InternalNode('c1')
    child2 = LinearLeafNode('c3', threshold=10)
    workload_info = {   # These parameters influence the benchmark workload
        'recordcount': 100000,
        'operationcount': 1000,
        'threadcount': 8
    }
    child2.update_workload_info(workload_info)
    root.add_children([child2], [1])
    # root.add_children([child1, child2], [1, 1])
    # child11 = LinearLeafNode('c11', threshold=5)
    # child12 = LinearLeafNode('c12', threshold=42)
    # child1.add_children([child11, child12], [1, 1])
    env = TreeEnvironment(root, 1)

    # Add workload information to clients ----------------------------------------------------------
    generate_workload_info(child2)
    # generate_propalloc_workload_info(child11)
    # generate_propalloc_workload_info(child12)
    return env


env = generate_env()
workload_exec = K8sWorkloadDeployer()

# Parse environment and create the workloads in kubernetes.
# This is equivalent to generating a YAML for each workload
# and running kubectl apply -f <yaml> for each workload.
workload_exec.deploy_environment(env)


# # DEBUG: get the generated outputs as json
# from kubernetes import client, config
# import json
# svc = workload_exec.parse_env_to_workloads(env)[0]
# with open('output.json', "w") as f:
#     json.dump(client.ApiClient().sanitize_for_serialization(svc), f, indent=4)