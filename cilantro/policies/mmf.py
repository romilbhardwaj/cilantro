"""
    Implements Max-min fairness.
    -- kirthevasank
"""


from copy import copy
import logging
import numpy as np
# Local
from cilantro.policies.base_policy import BasePolicy

logger = logging.getLogger(__name__)


def max_min_fairness(entitlements, demands, total_resource=1.0):
    """ Stand-alone function for max-min fairness. """
    resource_left = total_resource
    entitlement_left = 1
    allocs = [None for _ in range(len(entitlements))]
    sorted_idxs = list(np.argsort([dem/ent for (ent, dem) in zip(entitlements, demands)]))
    remaining_idxs = copy(sorted_idxs)
    for idx in sorted_idxs:
        if demands[idx] < resource_left * entitlements[idx] / entitlement_left:
            allocs[idx] = demands[idx]
            resource_left -= demands[idx]
            entitlement_left -= entitlements[idx]
            remaining_idxs.pop(0)
        else:
            for rem_idx in remaining_idxs:
                allocs[rem_idx] = resource_left * entitlements[rem_idx] / entitlement_left
            break
    assert sum(allocs) <= total_resource + 1e-5
    return allocs


class HMMF(BasePolicy):
    """ Hierarchical Max-min Fairness. """

    def _policy_initialise(self):
        """ Initialisation. """
        pass

    def _get_resource_allocation_for_loads(self, loads=None, to_normalise=False):
        """ Return allocations.
            If curr_loads is None, it will use the current load of the environment.
        """
        # pylint: disable=arguments-differ
        ret = {}
        if loads:
            for (leaf_path, ld) in loads.items():
                self.env.leaf_nodes[leaf_path].set_curr_load(ld)
        self.env.root.compute_curr_demand() # compute current demand bottom to top
        nodes_left_to_allocate = [(self.env.root, self.resource_quantity)]
        while nodes_left_to_allocate:
            curr_node, curr_resource = nodes_left_to_allocate.pop(0)
            if not hasattr(curr_node, 'children'):
                ret[curr_node.get_path_from_root()] = curr_resource
            else:
                child_demands = [curr_node.children[key][0].get_curr_demand()
                                 for key in curr_node.children]
                child_entitlements = [curr_node.children[key][2] for key in curr_node.children]
                child_allocs = max_min_fairness(child_entitlements, child_demands, curr_resource)
                child_nodes = [val[0] for _, val in curr_node.children.items()]
                nodes_left_to_allocate.extend(zip(child_nodes, child_allocs))
        if to_normalise:
            for key in ret:
                ret[key] = ret[key] / self.resource_quantity
        return ret


class HMMFDirect(HMMF):
    """ Hierarchical Max-min Fairness for directly allocating resources. """
    # pylint: disable=arguments-differ

    def _get_resource_allocation_for_loads(self, loads):
        """ Get resource allocation for loads. """
        if self.num_alloc_quanta: # If not None, we will have to allocate in discrete quanta
            allocs_normalised = super()._get_resource_allocation_for_loads(loads, to_normalise=True)
            entitlements = self.env.get_entitlements()
            # Now sort these allocations in ascending order of fair share allocations
            entitlements_as_list = list(entitlements.items())
            entitlements_as_list.sort(key=lambda x: x[1])
            num_leafs = len(entitlements_as_list)
            # Compute the discrete quanta
            ret = {}
            remainders = {}
            num_remaining_quanta = self.num_alloc_quanta
            for idx in range(num_leafs):
                leaf, leaf_entitlement = entitlements_as_list[idx]
                unnorm_alloc = allocs_normalised[leaf] * self.resource_quantity
                floor_alloc = int(np.floor(unnorm_alloc))
                ceil_alloc = int(np.ceil(unnorm_alloc))
                if ceil_alloc <= leaf_entitlement * self.resource_quantity:
                    ret[leaf] = ceil_alloc
                    diff_alloc_frac = ceil_alloc / self.resource_quantity - allocs_normalised[leaf]
                    for j in range(idx+1, num_leafs):
                        leaf_j, _ = entitlements_as_list[j]
                        allocs_normalised[leaf_j] -= diff_alloc_frac * allocs_normalised[leaf_j]
                else:
                    ret[leaf] = floor_alloc
                    remainders[leaf] = unnorm_alloc - ret[leaf]
            num_remaining_quanta = self.num_alloc_quanta - sum([val for _, val in ret.items()])
            # Now call randomised rounding
            if num_remaining_quanta > 0 and remainders:
                chosen_leafs = self._randomised_rounding(remainders, num_remaining_quanta)
                for leaf in chosen_leafs:
                    ret[leaf] += 1
            # Re-normalise
            ret_renorm = {key: val * self.alloc_granularity for key, val in ret.items()}
            allocs = ret_renorm
        else:
            allocs = super()._get_resource_allocation_for_loads(loads, to_normalise=False)
        return allocs

