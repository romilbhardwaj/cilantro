"""
    Generates the environment.
    -- kirthevasank
"""

from cilantro.core.henv import InternalNode, LinearLeafNode, TreeEnvironment

HOTELRES_MICROSERVICES = ['consul', 'frontend', 'geo', 'jaeger',
                          'memcached-profile', 'memcached-rate',
                          'memcached-reserve',
                          'mongodb-geo', 'mongodb-profile', 'mongodb-rate',
                          'mongodb-recommendation', 'mongodb-reservation',
                          'mongodb-user', 'profile', 'rate', 'recommendation',
                          'reservation', 'search', 'user']


def generate_env(env_descr, cluster_type, real_or_dummy='real'):
    """ Generates the environment.
        cluster_type is eks or kind.
    """
    if env_descr == 'hotelres':
        env = generate_hotelres_env(real_or_dummy)
    else:
        raise NotImplementedError(
            'Not implemented env_descr=%s yet.' % (env_descr))
    return env


def generate_hotelres_env(real_or_dummy):
    """ Generate hotel reservation environment. """
    job_dict = {ms: {'threshold': 1, 'util_scaling': 'linear', 'workload_type': real_or_dummy}
                for ms in HOTELRES_MICROSERVICES}
    # TODO(romilb): Not sure what all we need to add here
    return get_flat_environment_from_dict_of_jobs(job_dict, real_or_dummy)


def get_flat_environment_from_dict_of_jobs(job_dict, real_or_dummy):
    """ job_dict is a dicitonary of dictionaries. """
    for key, val in job_dict.items():
        if not ('name' in val):
            val['name'] = key  # Make sure you add the name to the dictionary.
    children_as_list = [_get_leaf_node_from_info_dict(val, real_or_dummy)
                        for _, val in job_dict.items()]
    weights = [1] * len(children_as_list)
    root = InternalNode('root')
    root.add_children(children_as_list, weights)
    env = TreeEnvironment(root, 1)
    return env


def _get_leaf_node_from_info_dict(info_dict, real_or_dummy):
    """ Returns a leaf node from a dictionary. """
    leaf = LinearLeafNode(info_dict['name'], threshold=info_dict['threshold'],
                          util_scaling=info_dict['util_scaling'])
    if real_or_dummy == 'dummy':
        workload_type = 'dummy' + info_dict['workload_type']
    elif real_or_dummy == 'real':
        workload_type = info_dict['workload_type']
    else:
        raise ValueError(
            'Unknown value for real_or_dummy: %s.' % (real_or_dummy))
    leaf.update_workload_info({'workload_type': workload_type})
    if 'workload_info' in info_dict:
        leaf.update_workload_info(info_dict['workload_info'])
    return leaf

