class BaseAllocSource(object):
    """
    Abstract class for allocation sources.
    AllocSources are used when the DataSource does not return an allocation.
    """
    def __init__(self):
        pass

    def get_allocation(self) -> int:
        raise NotImplementedError("Implement in a child class.")