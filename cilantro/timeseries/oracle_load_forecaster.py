"""
    An oracle which knows the load ahead of time.
    -- romilbhardwaj
    -- kirthevasank
"""

from cilantro.timeseries.ts_forecaster_bank import TSForecaster


class OracleLoadForecaster(TSForecaster):
    """ An oracle which knows the load ahead of time. """

    def __init__(self, env_leaf_node):
        """ constructor. """
        self.env_leaf_node = env_leaf_node

    def initialise(self):
        """ initialise. """
        pass

    def get_load_prediction(self, num_steps_ahead=None, conf_alpha=None):
        """ Obtain the load prediction. """
        curr_load = self.env_leaf_node.get_curr_load()
        return curr_load, curr_load, curr_load

