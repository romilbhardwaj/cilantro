"""
    Base data logger class.
    -- romilbhardwaj
    -- kirthevasank
"""

class BaseDataLogger:
    """ Base data logger. """

    def __init__(self):
        """ Constructor. """
        pass

    def log_event(self, event):
        """ Logs an event.
              :param event:
              :return:
        """
        raise NotImplementedError

    def get_data(self, fields, start_time_stamp=None, end_time_stamp=None):
        """ Request train samples. Returns a tuple. The first argument is a list of dictionaries.
            The second is the time stamp of the last data point.
        """
        raise NotImplementedError('Implement in a child class.')

