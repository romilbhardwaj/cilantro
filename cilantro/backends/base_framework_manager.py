"""
    Harness for cluster management.
    -- romilbhardwaj
    -- kirthevasank
"""

from typing import Dict


class BaseFrameworkManager:
    '''
    Manages interfacing with the resource management framework.
    Eg. calls to kubernetes to change resource allocation
    '''
    def __init__(self):
        pass

    def apply_allocation(self,
                         allocation: Dict[str, float]):
        """ Apply allocation. """
        raise NotImplementedError

    def get_cluster_resources(self,
                              resource_label: str = 'cpu'):
        """ Get amount of cluster resources. """
        raise NotImplementedError

    def get_alloc_granularity(self,
                              resource_label: str = 'cpu'):
        """ Return the granularity of allocation, which is the minimum quantum of resources
            we can allocate for a job.
        """
        raise NotImplementedError
