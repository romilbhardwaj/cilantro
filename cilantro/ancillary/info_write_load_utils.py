"""
    Some utilities to save and load information.
    -- romilbhardwaj
    -- kirthevasank
"""

import os
from cilantro.core.henv import load_env_from_file


ENV_FILE = 'env.txt'
INFO_FILE = 'info.txt'
INFO_DELIMITER = ':::'

REQD_INFO = ['alloc_granularity', 'resource_quantity']

def write_experiment_info_to_files(dir_name, env, info):
    """ Writes information to the file. """
    for reqinf in REQD_INFO:
        assert reqinf in info
    if not os.path.exists:
        os.makedirs(dir_name)
    env_file_path = os.path.join(dir_name, ENV_FILE)
    info_file_path = os.path.join(dir_name, INFO_FILE)
    env.write_to_file(env_file_path)
    info_str = '\n'.join(['%s%s%s'%(key, INFO_DELIMITER, val) for key, val in info.items()])
    with open(info_file_path, 'w') as info_file_handle:
        info_file_handle.write(info_str)
        info_file_handle.close()


def load_info_file(info_file_path):
    """ Loads an info file. """
    ret = {}
    with open(info_file_path, 'r') as file_handle:
        lines = [line.strip() for line in file_handle.readlines()]
        line_elems = [line.split(INFO_DELIMITER) for line in lines]
        for elem in line_elems:
            key = elem[0]
            val = elem[1]
            if key in ['alloc_granularity', 'resource_quantity']:
                val = int(float(val))
            ret[key] = val
    return ret


def read_experiment_info_from_files(dir_name):
    """ Reads experimental information from file. """
    env_file_path = os.path.join(dir_name, ENV_FILE)
    info_file_path = os.path.join(dir_name, INFO_FILE)
    env = load_env_from_file(env_file_path)
    info = load_info_file(info_file_path)
    return env, info

