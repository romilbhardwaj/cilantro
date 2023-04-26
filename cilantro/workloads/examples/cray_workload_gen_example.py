import logging

from cilantro.core.henv import InternalNode, LinearLeafNode, TreeEnvironment
from cilantro.workloads.cray_workload_generator import CRayWorkloadGenerator
from cilantro.workloads.k8s_workload_deployer import K8sWorkloadDeployer

logging.basicConfig(level=logging.DEBUG)


def generate_env():
    """ Generate a synthetic organisational tree. """
    workgen = CRayWorkloadGenerator(cilantro_image="public.ecr.aws/cilantro/cilantro:latest")

    def generate_workload_info(leaf: LinearLeafNode):
        [_, weight, _] = leaf.parent.children[leaf.name]
        path = leaf.get_path_from_root()
        workload_info = leaf.workload_info
        workload_server_objs = workgen.generate_workload_server_objects(app_name=path,
                                                                        threshold=leaf.threshold,
                                                                        app_weight=weight,
                                                                        app_unit_demand=leaf.unit_demand,
                                                                        **workload_info)

        # Generate workload client. For the cray workload, the cilantro client is
        # embedded with the workload client. In other words, the cilantro client is a
        # part of workload_client_objs here.

        workload_client_objs = workgen.generate_workload_client_objects(app_name=path,
                                                                        threshold=leaf.threshold,
                                                                        **workload_info)

        # generate_cilantro_client_objects returns empty list for cray workload
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

    # ====== WORKLOAD INFO FOR CRAY WORKLOADS ==========
    # Workload_info for cray workloads has two sub-dictionaries: cray_client_override_args
    # and cilantro_client_override_args. One is for the Cray workload client, and the
    # other is for args to the cilantro client (cray_to_grpc_driver.py).

    # ====== For cray_client_override_args ========
    # --cray-utilfreq: Controls the frequency of utility reports (seconds). Too low and tasks won't complete, too long and new resource allocations will be unutilized for longer.
    # --cray-workload-type: Selects workload to run. Either of sleep_task, db, modserve or learning.
    # --sleep-time: Duration of the sleep task - reducing this causes tasks to complete sooner.
    # --trace-scalefactor: Scaling factor for trace load

    workload_info = {
        'cray_client_override_args': {"--cray-utilfreq": "10",
                                      "--cray-workload-type": "sleep_task",
                                      "--sleep-time": "0.2",
                                      "--trace-scalefactor": "2"},

        # Args to cilantro client. Passed onto cray_to_grpc_driver.py.
        'cilantro_client_override_args': {"--slo-type": "latency",
                                          "--slo-latency": "1",
                                          "--max-throughput": "-1"}
    }
    child2.update_workload_info(workload_info)
    root.add_children([child2], [1])
    env = TreeEnvironment(root, 1)

    # Add workload information to clients ----------------------------------------------------------
    generate_workload_info(child2)
    return env


env = generate_env()
workload_exec = K8sWorkloadDeployer()

# Parse environment and create the workloads in kubernetes.
# This is equivalent to generating a YAML for each workload
# and running kubectl apply -f <yaml> for each workload.
# workload_exec.parse_env_to_workloads(env)
workload_exec.deploy_environment(env)
