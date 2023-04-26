"""
    Generate the environment for the experiment
    -- romilbhardwaj
    -- kirthevasank
"""

# pylint: disable=too-many-branches

# From cilantro
from cilantro.core.henv import InternalNode, LinearLeafNode, TreeEnvironment
from cilantro.workloads.cassandrahorz_workload_generator import CassandraHorzWorkloadGenerator
from cilantro.workloads.cray_workload_generator import CRayWorkloadGenerator
# In Demo
from dummy.dummy_workload_generator import DummyWorkloadGenerator



def generate_env(env_descr, cluster_type, real_or_dummy):
    """ Generate a synthetic organisational tree. """
    # Create the environment -----------------------------------------------------------------------
    if env_descr == 'craysleeptest':
        env = generate_env_cray_sleep_test(real_or_dummy)
    elif env_descr == 'craypredservtest':
        env = generate_env_cray_predserv_test(real_or_dummy)
    elif env_descr == 'craymltraintest':
        env = generate_env_cray_mltrain_test(real_or_dummy)
    elif env_descr == 'craydbtest':
        env = generate_env_cray_db_test(real_or_dummy)
    elif env_descr == 'profiling_env_1':
        env = generate_env_profiling_env1(real_or_dummy)
    elif env_descr == 'exp_env_1':
        env = generate_env_exp_env_1(real_or_dummy)
    elif env_descr == 'as_db1':
        env = generate_as_db1(real_or_dummy)
    elif env_descr == 'as_prs':
        env = generate_as_prs(real_or_dummy)
    elif env_descr == 'test':
        env = generate_env_test(real_or_dummy)
    elif env_descr == 'simple1':
        env = generate_env_simple1(real_or_dummy)
    elif env_descr == 'flat1':
        env = generate_env_flat1(real_or_dummy)
    elif env_descr == 'flat2':
        env = generate_env_flat2(real_or_dummy)
    elif env_descr == 'flat3':
        env = generate_env_flat3(real_or_dummy)
    # Autoscaling environments ---------------------------------------------------------------------
    elif env_descr == 'asds1':
        env = generate_env_asds1(real_or_dummy)
    elif env_descr == 'asds2':
        env = generate_env_asds2(real_or_dummy)
    elif env_descr == 'asws1':
        env = generate_env_asws1(real_or_dummy)
    elif env_descr == 'asws2':
        env = generate_env_asws2(real_or_dummy)
    elif env_descr == 'asim1':
        env = generate_env_asim1(real_or_dummy)
    elif env_descr == 'asim2':
        env = generate_env_asim2(real_or_dummy)
    else:
        raise NotImplementedError('Not implemented env_descr=%s yet.'%(env_descr))
    # Create workload info -------------------------------------------------------------------------
    generate_workload_info_for_environment(env, cluster_type)
    return env

def get_workload_generator(workload_type: str):
    """ Returns workload generator. """
    if 'dummy' in workload_type:
        return DummyWorkloadGenerator
    elif workload_type == 'dataserving':
        return CassandraHorzWorkloadGenerator
    elif workload_type.startswith('cray'):
        return CRayWorkloadGenerator
    else:
        raise NotImplementedError("Workload type %s not implemented."%(workload_type))

def generate_workload_info_for_environment(env, cluster_type):
    """ Generates workload information from an environment. """
    # Create k8s objects for each leaf =========================================================
    for leaf_path, leaf in env.leaf_nodes.items():
        workload_type = leaf.get_workload_info('workload_type')
        workload_info = leaf.workload_info
        workload_gen_constructor = get_workload_generator(workload_type)
        workgen = workload_gen_constructor(cluster_type=cluster_type)
        [_, weight, _] = leaf.parent.children[leaf.name]
        workload_server_objs = workgen.generate_workload_server_objects(
            app_name=leaf_path, threshold=leaf.threshold, app_weight=weight,
            app_unit_demand=leaf.unit_demand, **workload_info)
        workload_workload_client_objs = workgen.generate_workload_client_objects(
            app_name=leaf_path, threshold=leaf.threshold, **workload_info)
        workload_cilantro_client_objs = workgen.generate_cilantro_client_objects(
            app_name=leaf_path, threshold=leaf.threshold, **workload_info)
        k8s_objects = [*workload_server_objs,
                       *workload_workload_client_objs,
                       *workload_cilantro_client_objs]
        leaf.update_workload_info({"k8s_objects": k8s_objects})


def _get_leaf_node_from_info_dict(info_dict, real_or_dummy):
    """ Returns a leaf node from a dictionary. """
    leaf = LinearLeafNode(info_dict['name'], threshold=info_dict['threshold'],
                          util_scaling=info_dict['util_scaling'])
    if real_or_dummy == 'dummy':
        workload_type = 'dummy' + info_dict['workload_type']
    elif real_or_dummy == 'real':
        workload_type = info_dict['workload_type']
    else:
        raise ValueError('Unknown value for real_or_dummy: %s.'%(real_or_dummy))
    leaf.update_workload_info({'workload_type': workload_type})
    if 'workload_info' in info_dict:
        leaf.update_workload_info(info_dict['workload_info'])
    return leaf


# Create basic nodes ray -------------------------------------------------------------------------
def get_craysleep_node(node_name, slo_latency, slo_thresh, slo_util_scaling, real_or_dummy):
    """ A simple environment with just one job. """
    # ====== WORKLOAD INFO FOR CRAY WORKLOADS ==========
    # Workload_info for cray workloads has two sub-dictionaries: cray_client_override_args
    # and cilantro_client_override_args. One is for the Cray workload client, and the
    # other is for args to the cilantro client (cray_to_grpc_driver.py).

    # ====== For cray_client_override_args ========
    # --cray-utilfreq: Controls the frequency of utility reports (seconds). Too low and tasks won't
    #   complete, too long and new resource allocations will be unutilized for longer.
    # --cray-workload-type: Selects workload to run. Either of sleep_task, db, modserve or learning.
    # --sleep-time: Duration of the sleep task - reducing this causes tasks to complete sooner.
    # --trace-scalefactor: Scaling factor for trace load
    workload_info = {
        'cray_client_override_args': {
            "--cray-utilfreq": "10",
            "--cray-workload-type": "sleep_task",
            "--sleep-time": "0.1",
            "--trace-path": "/cray_workloads/traces/twit-b1000-n88600.csv",
            "--trace-scalefactor": "1.0"},

        # Args to cilantro client. Passed onto cray_to_grpc_driver.py.
        'cilantro_client_override_args': {"--slo-type": "latency",
                                          "--slo-latency": str(slo_latency),
                                          "--max-throughput": "-1"}
    }
    child_info = {'name': node_name, 'threshold': slo_thresh, 'workload_type': 'craysleep',
                  'util_scaling': slo_util_scaling, 'workload_info': workload_info}
    return _get_leaf_node_from_info_dict(child_info, real_or_dummy)


def get_craypredserv_node(node_name, slo_latency, slo_thresh, slo_util_scaling, real_or_dummy):
    """ Returns a prediction serving node. """
    workload_info = {
        'cray_client_override_args': {
            "--cray-utilfreq": "10",
            "--cray-workload-type": "predserv_task",
            "--serve-chunk-size": "4",
            "--sleep-time": "0.0",
            "--trace-scalefactor": "1.0",
            "--trace-path": "/cray_workloads/traces/twit-b1000-n88600.csv",
            "--ps-data-path": "/cray_workloads/train_data/news_popularity.p",
            "--ps-model-path": "/cray_workloads/train_data/news_rfr.p",
         },
        # Args to cilantro client. Passed onto cray_to_grpc_driver.py.
        'cilantro_client_override_args': {"--slo-type": "latency",
                                          "--slo-latency": str(slo_latency),
                                          "--max-throughput": "-1"}
    }
    child_info = {'name': node_name, 'threshold': slo_thresh, 'workload_type': 'craypredserv',
                  'util_scaling': slo_util_scaling, 'workload_info': workload_info}
    return _get_leaf_node_from_info_dict(child_info, real_or_dummy)


def get_craymltrain_node(node_name, slo_thresh, slo_util_scaling, real_or_dummy):
    """ Returns a ML train node. """
    workload_info = {
        'cray_client_override_args': {
            "--cray-utilfreq": "10",
            "--cray-workload-type": "mltrain_task",
            "--sleep-time": "0.0",
            # ml train params
            "--train-data-path": "/cray_workloads/train_data/naval_propulsion.p",
            "--train-batch-size": "16",
            "--train-num-iters": "10",
            },

        # Args to cilantro client. Passed onto cray_to_grpc_driver.py.
        'cilantro_client_override_args': {"--slo-type": "throughput",
                                          "--slo-latency": "2",
                                          "--max-throughput": "-1"}
    }
    child_info = {'name': node_name, 'threshold': slo_thresh, 'workload_type': 'craymltrain',
                  'util_scaling': slo_util_scaling, 'workload_info': workload_info}
    return _get_leaf_node_from_info_dict(child_info, real_or_dummy)


def get_craydb_node(node_name, bin_num, sleep_time, trace_scale_factor, slo_latency, slo_thresh,
                    slo_util_scaling, real_or_dummy):
    """ Returns a DB node. """
    workload_info = {
        'cray_client_override_args': {
            "--cray-utilfreq": "10",
            "--cray-workload-type": "db_task",
            "--sleep-time": str(sleep_time),
            "--trace-scalefactor": str(trace_scale_factor),
            "--trace-path": "/cray_workloads/traces/twit-b1000-n88600.csv",
            "--query-bin": str(bin_num),
            "--db-path": "/cray_workloads/db_data/tpcds_data/sqlite/tpcds.db",
            "--queries-file-path": "/cray_workloads/db_data/tpcds_data/queries/processed",
            "--query-bins-path": "/cray_workloads/db_bins/bins_kk_duplicate.json",
         },

        # Args to cilantro client. Passed onto cray_to_grpc_driver.py.
        'cilantro_client_override_args': {"--slo-type": "latency",
                                          "--slo-latency": str(slo_latency),
                                          "--max-throughput": "-1"}
    }
    workload_type = 'craydb' + str(bin_num)
    child_info = {'name': node_name, 'threshold': slo_thresh, 'workload_type': workload_type,
                  'util_scaling': slo_util_scaling, 'workload_info': workload_info}
    return _get_leaf_node_from_info_dict(child_info, real_or_dummy)

def get_craydb0_node(node_name, *args, **kwargs):
    """ db node with bin 0. """
    return get_craydb_node(node_name, 0, 0.0, 0.5, *args, **kwargs)

# def get_craydb2_node(node_name, *args, **kwargs):
#     """ db node with bin 0. """
#     return get_craydb_node(node_name, 1, 0.00, 1.05, *args, **kwargs)

def get_craydb1_node(node_name, *args, **kwargs):
    """ db node with bin 0. """
    return get_craydb_node(node_name, 1, 0.0, 1.05, *args, **kwargs)

# Environments for profiling --------------------------------------------------------------------
def generate_env_cray_sleep_test(real_or_dummy):
    """ A simple environment with just one job. """
    root = InternalNode('root')
    child1 = get_craysleep_node('j01', 2, 0.9, 'linear', real_or_dummy)
    root.add_children([child1], [1])
    env = TreeEnvironment(root, 1)
    return env

def generate_env_cray_predserv_test(real_or_dummy):
    """ A simple environment with just one prediction serving job. """
    child1 = get_craypredserv_node('j01', 2, 0.9, 'linear', real_or_dummy)
    root = InternalNode('root')
    root.add_children([child1], [1])
    env = TreeEnvironment(root, 1)
    return env

def generate_env_cray_mltrain_test(real_or_dummy):
    """ A simple environment with just one prediction serving job. """
    root = InternalNode('root')
    child1 = get_craymltrain_node('j01', 200, 'linear', real_or_dummy)
    root.add_children([child1], [1])
    env = TreeEnvironment(root, 1)
    return env

def generate_env_cray_db_test(real_or_dummy):
    """ A simple environment with just one prediction serving job. """
    root = InternalNode('root')
    child1 = get_craydb0_node('j01', 4, 0.9, 'linear', real_or_dummy)
    root.add_children([child1], [1])
    env = TreeEnvironment(root, 1)
    return env

def generate_env_profiling_env1(real_or_dummy):
    """ Creates an environment for profiling. """
    children_list = [
        get_craydb0_node('j01', slo_latency=2, slo_thresh=0.9, slo_util_scaling='linear',
                         real_or_dummy=real_or_dummy),
        get_craydb1_node('j02', slo_latency=2, slo_thresh=0.9, slo_util_scaling='linear',
                         real_or_dummy=real_or_dummy),
        get_craymltrain_node('j03', slo_thresh=200, slo_util_scaling='linear',
                             real_or_dummy=real_or_dummy),
        get_craypredserv_node('j04', slo_latency=2, slo_thresh=0.9, slo_util_scaling='linear',
                              real_or_dummy=real_or_dummy),
        ]
    weights = [1] * len(children_list)
    root = InternalNode('root')
    root.add_children(children_list, weights)
    env = TreeEnvironment(root, 1)
    return env


def generate_env_exp_env_template(real_or_dummy, db0_slos, db1_slos, mlt_slos, prs_slos):
    """ Generates template environment. """
    children_list = []
    prefix_slo_dict = {
        'db0': (db0_slos,
                lambda name, thresh, scaling: get_craydb0_node(node_name=name,
                                                               slo_latency=2,
                                                               slo_thresh=thresh,
                                                               slo_util_scaling=scaling,
                                                               real_or_dummy=real_or_dummy)),
        'db1': (db1_slos,
                lambda name, thresh, scaling: get_craydb1_node(node_name=name,
                                                               slo_latency=2,
                                                               slo_thresh=thresh,
                                                               slo_util_scaling=scaling,
                                                               real_or_dummy=real_or_dummy)),
        'mlt': (mlt_slos,
                lambda name, thresh, scaling: get_craymltrain_node(node_name=name,
                                                                   slo_thresh=thresh,
                                                                   slo_util_scaling=scaling,
                                                                   real_or_dummy=real_or_dummy)),
        'prs': (prs_slos,
                lambda name, thresh, scaling: get_craypredserv_node(node_name=name, slo_latency=2,
                                                                    slo_thresh=thresh,
                                                                    slo_util_scaling=scaling,
                                                                    real_or_dummy=real_or_dummy)),
        }
    # Crate list of children ---------------------------------------------------------
    for key, (slo_list, job_constructor) in prefix_slo_dict.items():
        key_counter = 0
        for slo_val, slo_util_scaling in slo_list:
            key_counter += 1
            job_name = key + 'j' + str(key_counter)
            job_node = job_constructor(job_name, slo_val, slo_util_scaling)
            children_list.append(job_node)
    # Create environment -------------------------------------------------------------
    weights = [1] * len(children_list)
    root = InternalNode('root')
    root.add_children(children_list, weights)
    env = TreeEnvironment(root, 1)
    return env


def generate_env_exp_env_1(real_or_dummy):
    """ Creates experimental environment 1. """
#     db0_slos = [0.9, 0.9, 0.95]
#     db1_slos = [0.9, 0.9, 0.95, 0.95, 0.95, 0.99, 0.99]
#     mlt_slos = [400, 400, 450, 450, 500, 500, 500]
#     prs_slos = [0.9, 0.9, 0.95]
    db0_slos = [(0.9, 'linear'),
                (0.9, 'linear'),
                (0.95, 'sqrt')]
    mlt_slos = [(400, 'sqrt'),
                (400, 'sqrt'),
                (450, 'linear'),
                (450, 'linear'),
                (500, 'quadratic'),
                (500, 'quadratic'),
                (500, 'quadratic')]
    db1_slos = [(0.9, 'linear'),
                (0.9, 'quadratic'),
                (0.95, 'quadratic'),
                (0.95, 'linear'),
                (0.95, 'quadratic'),
                (0.99, 'quadratic'),
                (0.99, 'sqrt')]
    prs_slos = [(0.9, 'linear'),
                (0.9, 'sqrt'),
                (0.95, 'sqrt')]
    return generate_env_exp_env_template(real_or_dummy, db0_slos, db1_slos, mlt_slos, prs_slos)


# auto-scaling environments -------------------------------------------------------------------
def generate_as_db1(real_or_dummy):
    """ Autoscaling environment with a single data serving job. """
    db0_slos = []
    db1_slos = [0.95]
    mlt_slos = []
    prs_slos = []
    return generate_env_exp_env_template(real_or_dummy, db0_slos, db1_slos, mlt_slos, prs_slos)

def generate_as_prs(real_or_dummy):
    """ Autoscaling environment with a single data serving job. """
    db0_slos = []
    db1_slos = []
    mlt_slos = []
    prs_slos = [0.95]
    return generate_env_exp_env_template(real_or_dummy, db0_slos, db1_slos, mlt_slos, prs_slos)


# Environments for experimenting ----------------------------------------------------------------
def generate_env_test(real_or_dummy):
    """ A simple environment with just one job. """
    # Set thresholds ------------------------------------------
    # Recordcount in cassandra controls the size of the DB,
    # Operationcount is the number of operations (queries) the benchmark sends
    # Threadcount is the number of threads used by the benchmark to parallelize sending operations
    # Threshold is latency slo in us
    c1_info = {'name': 'j01', 'threshold': 0.9, 'workload_type': 'dataserving',
               'workload_info': {
                   'recordcount': 10000000,
                   'operationcount': 10000,
                   'threadcount': 10,
                   'slo_latency': 1000,
                   'client_target_throughput': 1000000,
               }}
    # Create environment --------------------------------------
    root = InternalNode('root')
    child1 = _get_leaf_node_from_info_dict(c1_info, real_or_dummy)
    root.add_children([child1], [1])
    env = TreeEnvironment(root, 1)
    return env

def generate_env_simple1(real_or_dummy):
    """ A simple environment. """
    # Set thresholds ------------------------------------------
    c2_info = {'name': 'c2', 'threshold': 0.80, 'workload_type': 'dataserving'}
    c11_info = {'name': 'c11', 'threshold': 0.99, 'workload_type': 'datacaching'}
    c12_info = {'name': 'c12', 'threshold': 0.95, 'workload_type': 'inmemoryanalytics'}
    c31_info = {'name': 'c31', 'threshold': 0.90, 'workload_type': 'webserving'}
    # Create environment --------------------------------------
    root = InternalNode('root')
    child1 = InternalNode('c1')
    child2 = _get_leaf_node_from_info_dict(c2_info, real_or_dummy)
    child3 = InternalNode('c3')
    root.add_children([child1, child2, child3], [1, 1, 1])
    child11 = _get_leaf_node_from_info_dict(c11_info, real_or_dummy)
    child12 = _get_leaf_node_from_info_dict(c12_info, real_or_dummy)
    child1.add_children([child11, child12], [2, 1])
    child31 = _get_leaf_node_from_info_dict(c31_info, real_or_dummy)
    child3.add_children([child31], [1])
    env = TreeEnvironment(root, 1)
    return env


def get_flat_environment_from_dict_of_jobs(job_dict, real_or_dummy):
    """ job_dict is a dicitonary of dictionaries. """
    for key, val in job_dict.items():
        if not ('name' in val):
            val['name'] = key # Make sure you add the name to the dictionary.
    children_as_list = [_get_leaf_node_from_info_dict(val, real_or_dummy)
                        for _, val in job_dict.items()]
    weights = [1] * len(children_as_list)
    root = InternalNode('root')
    root.add_children(children_as_list, weights)
    env = TreeEnvironment(root, 1)
    return env


def generate_env_flat1(real_or_dummy):
    """ Creates a flat environment. """
    job_dict = {}
    job_dict['j01'] = {'threshold': 0.8, 'workload_type': 'dataserving'}
    job_dict['j02'] = {'threshold': 0.9, 'workload_type': 'dataserving'}
    job_dict['j03'] = {'threshold': 0.95, 'workload_type': 'dataserving'}
    job_dict['j04'] = {'threshold': 0.8, 'workload_type': 'datacaching'}
    job_dict['j05'] = {'threshold': 0.9, 'workload_type': 'datacaching'}
    job_dict['j06'] = {'threshold': 0.95, 'workload_type': 'datacaching'}
    job_dict['j07'] = {'threshold': 0.8, 'workload_type': 'inmemoryanalytics'}
    job_dict['j08'] = {'threshold': 0.9, 'workload_type': 'inmemoryanalytics'}
    job_dict['j09'] = {'threshold': 0.95, 'workload_type': 'inmemoryanalytics'}
    job_dict['j10'] = {'threshold': 0.8, 'workload_type': 'webserving'}
    job_dict['j11'] = {'threshold': 0.9, 'workload_type': 'webserving'}
    job_dict['j12'] = {'threshold': 0.95, 'workload_type': 'webserving'}
    job_dict['j13'] = {'threshold': 0.8, 'workload_type': 'dataanalytics'}
    job_dict['j14'] = {'threshold': 0.9, 'workload_type': 'dataanalytics'}
    job_dict['j15'] = {'threshold': 0.95, 'workload_type': 'dataanalytics'}
    job_dict['j16'] = {'threshold': 0.8, 'workload_type': 'websearch'}
    job_dict['j17'] = {'threshold': 0.9, 'workload_type': 'websearch'}
    job_dict['j18'] = {'threshold': 0.95, 'workload_type': 'websearch'}
    return get_flat_environment_from_dict_of_jobs(job_dict, real_or_dummy)


def generate_flat_env_from_workload_to_thresh_dict(workload_to_thresh_val_dict, real_or_dummy,
                                                   num_users_per_config):
    """ Generates a flat environment from a workload to thresh val dictionary. """
    job_counter = 0
    job_dict = {}
    for wlt, thresh_vals in workload_to_thresh_val_dict.items():
        for thresh in thresh_vals:
            for _ in range(num_users_per_config):
                job_counter += 1
                job_key = 'j%02d'%(job_counter)
                job_dict[job_key] = {'threshold': thresh, 'workload_type': wlt}
    return get_flat_environment_from_dict_of_jobs(job_dict, real_or_dummy)


def generate_env_flat2(real_or_dummy):
    """ Generate environment flat 2. """
    latency_thresh_vals = [0.8, 0.85, 0.9, 0.95, 0.975, 0.99]
    linear_thresh_vals = [0.8, 0.95, 0.99, 1.05, 1.3, 1.5]
    workload_types = ['dataserving', 'datacaching', 'inmemoryanalytics', 'webserving',
                      'dataanalytics', 'websearch']
    workload_to_thresh_val_dict = {}
    for wlt in workload_types:
        workload_to_thresh_val_dict[wlt] = \
            linear_thresh_vals if wlt in ['webserving'] else latency_thresh_vals
    return generate_flat_env_from_workload_to_thresh_dict(workload_to_thresh_val_dict,
                                                          real_or_dummy, 2)


def generate_env_flat3(real_or_dummy):
    """ Generate environment flat 2. """
    job_dict = {}
    job_dict['j01'] = {'threshold': 0.5, 'workload_type': 'dataserving'}
    job_dict['j02'] = {'threshold': 0.5, 'workload_type': 'dataserving'}
    job_dict['j03'] = {'threshold': 0.5, 'workload_type': 'dataserving'}
    job_dict['j04'] = {'threshold': 0.5, 'workload_type': 'dataserving'}

    job_dict['j06'] = {'threshold': 0.9, 'workload_type': 'datacaching'}
    job_dict['j07'] = {'threshold': 0.95, 'workload_type': 'datacaching'}
    job_dict['j08'] = {'threshold': 0.95, 'workload_type': 'datacaching'}

    job_dict['j11'] = {'threshold': 0.9, 'workload_type': 'inmemoryanalytics'}
    job_dict['j12'] = {'threshold': 0.95, 'workload_type': 'inmemoryanalytics'}
    job_dict['j13'] = {'threshold': 0.99, 'workload_type': 'inmemoryanalytics'}

    job_dict['j16'] = {'threshold': 0.5, 'workload_type': 'dataanalytics'}
    job_dict['j17'] = {'threshold': 0.5, 'workload_type': 'dataanalytics'}
    job_dict['j18'] = {'threshold': 0.55, 'workload_type': 'dataanalytics'}

    job_dict['j21'] = {'threshold': 0.5, 'workload_type': 'websearch'}
    job_dict['j22'] = {'threshold': 0.5, 'workload_type': 'websearch'}
    job_dict['j23'] = {'threshold': 0.5, 'workload_type': 'websearch'}
    return get_flat_environment_from_dict_of_jobs(job_dict, real_or_dummy)


# Autoscaling environments ------------------------------------------------------------------
def generate_env_asds1(real_or_dummy):
    """ Autoscaling environment with a single data serving job. """
    job_dict = {'j01': {'threshold': 0.95, 'workload_type': 'dataserving'}}
    return get_flat_environment_from_dict_of_jobs(job_dict, real_or_dummy)

def generate_env_asds2(real_or_dummy):
    """ Autoscaling environment with a single data serving job. """
    job_dict = {'j01': {'threshold': 0.99, 'workload_type': 'dataserving'}}
    return get_flat_environment_from_dict_of_jobs(job_dict, real_or_dummy)

def generate_env_asws1(real_or_dummy):
    """ Autoscaling environment with a single data serving job. """
    job_dict = {'j01': {'threshold': 0.95, 'workload_type': 'websearch'}}
    return get_flat_environment_from_dict_of_jobs(job_dict, real_or_dummy)

def generate_env_asws2(real_or_dummy):
    """ Autoscaling environment with a single data serving job. """
    job_dict = {'j01': {'threshold': 0.99, 'workload_type': 'websearch'}}
    return get_flat_environment_from_dict_of_jobs(job_dict, real_or_dummy)

def generate_env_asim1(real_or_dummy):
    """ Autoscaling environment with a single data serving job. """
    job_dict = {'j01': {'threshold': 0.95, 'workload_type': 'inmemoryanalytics'}}
    return get_flat_environment_from_dict_of_jobs(job_dict, real_or_dummy)

def generate_env_asim2(real_or_dummy):
    """ Autoscaling environment with a single data serving job. """
    job_dict = {'j01': {'threshold': 0.999, 'workload_type': 'inmemoryanalytics'}}
    return get_flat_environment_from_dict_of_jobs(job_dict, real_or_dummy)

