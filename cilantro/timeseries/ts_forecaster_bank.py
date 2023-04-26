"""
    A bank for forecasters.
    -- kirthevasank
    -- romilbhardwaj
"""

# from cilantro.core.bank import Bank
from cilantro.learners.learner_bank import LearnerBank


class TSForecaster:
    """ An abstraction for time series forecasting. """

    def initialise(self):
        """ Initialise. """
        raise NotImplementedError('Implement in a child class.')

    def stop_training_loop(self):
        """ initialise. """
        pass

    def forecast(self, num_steps_ahead, conf_alpha):
        """ Forecasts next element in time series. """
        raise NotImplementedError('Implement in a child class.')


class TSForecasterBank(LearnerBank):
    """ Time series forecaster Bank. """

    @classmethod
    def _check_type(cls, obj):
        """ Checks type. """
        assert isinstance(obj, TSForecaster)

