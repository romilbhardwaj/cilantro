"""
    Profiled information loader.
    -- kirthevasank
    -- romilbhardwaj
"""

import logging
import os
import pickle
import numpy as np
# Local
from cilantro.core.bank import Bank


logger = logging.getLogger(__name__)


def _load_profiled_data_from_file(file_path):
    """ Loads profiled data from file. """
    with open(file_path, 'rb') as pickle_file:
        profiled_data = pickle.load(pickle_file)
        pickle_file.close()
    return profiled_data


class ProfiledInfo:
    """ Class for loading profiled information for a workload type. """

    def __init__(self, workload_type, profiled_data, file_path=None):
        """ Constructor. """
        self.workload_type = workload_type
        if profiled_data is None:
            assert isinstance(file_path, str)
            self.profiled_data = _load_profiled_data_from_file(file_path)
        else:
            self.profiled_data = profiled_data

    def get_alloc_for_payoff_vals(self, payoff_vals, est_type='est'):
        """ Returns the alloc value for the given threshold.
            use_est should be one of ucbs, lcbs, or ests, indicating if we wish a upper confidence
            bound, lower confidence bound or an estimate for the alloc.
        """
        interp_Y = self.profiled_data['grid']
        if est_type == 'est':
            interp_X = self.profiled_data['ests']
        elif est_type == 'ucb': # If we want a UCB on the alloc use an LCB of the estimated payoff
            interp_X = self.profiled_data['lcbs']
        elif est_type == 'lcb': # If we want an LCB on the alloc use a UCB of the estimated payoff
            interp_X = self.profiled_data['ucbs']
        else:
            raise ValueError('Unknown est_type %s.'%(est_type))
        interp = np.interp(payoff_vals, interp_X, interp_Y)
        return interp

    def get_payoff_for_alloc_vals(self, alloc_vals, est_type='est'):
        """ Returns a payoff for the alloc. """
        interp_X = self.profiled_data['grid']
        interp_Y = self.profiled_data[est_type + 's']
        interp = np.interp(alloc_vals, interp_X, interp_Y)
        return interp

    def get_load_quantiles(self):
        """ Returns the p50 and p99 load quantiles. """
        p50_load = self.profiled_data['p50_load'] if 'p50_load' in self.profiled_data else -1
        p99_load = self.profiled_data['p99_load'] if 'p99_load' in self.profiled_data else -1
        return p50_load, p99_load


class ProfiledInfoBank (Bank):
    """ Bank for Profiled Information Loaders. """

    def __init__(self, profiled_info_dir=None):
        """ Constructor. """
        super().__init__()
        if profiled_info_dir:
            self.load_profiled_info_from_dir(profiled_info_dir)

    def load_profiled_info_from_dir(self, profiled_info_dir):
        """ Loads profiled information from a directory.
            profiled_info_dir is the directory name and has a list of pickled files with names
            '{workload_type}.p'.
        """
        prof_files = [elem for elem in os.listdir(profiled_info_dir) if elem.endswith('.p')]
        for pf in prof_files:
            pickle_file_path = os.path.join(profiled_info_dir, pf)
            profiled_data = _load_profiled_data_from_file(pickle_file_path)
            workload_type = profiled_data['workload_type']
            profiled_info_loader = ProfiledInfo(workload_type, profiled_data=profiled_data)
            self.register(workload_type, profiled_info_loader)

    @classmethod
    def _check_type(cls, obj):
        """ Checks the type of the object. """
        return isinstance(obj, ProfiledInfo)

    def get_unit_demand_for_leaf_node(self, leaf_node, est_type='est'):
        """ Returns the unit demand for the leaf node. """
        workload_type = leaf_node.get_workload_info('workload_type')
        threshold = leaf_node.threshold
        dem_list = self.get(workload_type).get_alloc_for_payoff_vals([threshold], est_type)
        return dem_list[0]

    def get_info_for_leaf_node(self, leaf_node, key):
        """ Obtain information for leaf node. """
        workload_type = leaf_node.get_workload_info('workload_type')
        return self.get(workload_type).profiled_data[key]

    def display_profiled_information_for_env(self, env, est_type='est'):
        """ Returns a string which displayes profiled information for the environment. """
        ret_p50_list = []
        est_p50_resources = 0
        ret_p99_list = []
        est_p99_resources = 0
        for leaf_path, leaf_node in env.leaf_nodes.items():
            workload_type = leaf_node.get_workload_info('workload_type')
            logger.info('workload_type: %s, %s', workload_type, self.get(workload_type))
            leaf_unit_demand = self.get(workload_type).get_alloc_for_payoff_vals(
                [leaf_node.threshold], est_type)[0]
            leaf_p50_load, leaf_p99_load = self.get(workload_type).get_load_quantiles()
            leaf_p50_demand = leaf_p50_load * leaf_unit_demand
            leaf_p99_demand = leaf_p99_load * leaf_unit_demand
            est_p50_resources += int(np.ceil(leaf_p50_demand))
            est_p99_resources += int(np.ceil(leaf_p99_demand))
            ret_p50_list.append('%s (%s): %0.3f  (ud=%0.5f, p50_load=%0.2f)'%(
                leaf_path, workload_type, leaf_p50_demand, leaf_unit_demand, leaf_p50_load))
            ret_p99_list.append('%s (%s): %0.3f  (ud=%0.5f, p99_load=%0.2f)'%(
                leaf_path, workload_type, leaf_p99_demand, leaf_unit_demand, leaf_p99_load))
        ret_str = '\n Estimated p50 total demand = %0.3f\n'%(est_p50_resources) + \
                  '\n'.join(ret_p50_list) + '\n' + \
                  'Estimated p99 total demand = %0.3f\n'%(est_p99_resources) + \
                  '\n'.join(ret_p99_list)
        return ret_str

