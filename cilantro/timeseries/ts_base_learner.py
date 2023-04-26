"""
    A module for learning time series models.
    -- kirthevasank
    -- romilbhardwaj
"""

from copy import deepcopy
import logging
# Local
from cilantro.timeseries.ts_forecaster_bank import TSForecaster

logger = logging.getLogger(__name__)

DFLT_SLEEP_TIME_BETWEEN_DATA_REPOLLS = 0.0

class TSModel:
    """ Time series model. """

    def __init__(self, name):
        """ Constructor. """
        self.name = name
        self._model_initialised = False

    # Add/process data -----------------------------------------------------------------------------
    def update_model_with_new_data(self, X):
        """ Updates model with the given data. """
        raise NotImplementedError('Implement in a child class.')

    def initialise_model(self):
        """ Initialise model. """
        if not self._model_initialised:
            self._initialise_model_child()
            self._model_initialised = True

    def _initialise_model_child(self):
        """ Initialise model. """
        raise NotImplementedError('Implement in a child class.')

    # Forecasting ---------------------------------------------------------------------------------
    def forecast(self, num_steps_ahead, conf_alpha):
        """ Forecast TS model. """
        raise NotImplementedError('Implement in a child class.')


class TSBaseLearner(TSForecaster):
    """ Base learner for time series models. """

    def __init__(self, app_id, data_logger, model, field_to_forecast,
                 update_on_copy_of_live_model=True):
        """ Constructor. """
        self.app_id = app_id
        # Obtain model --------------------------------------------------------
        if isinstance(model, TSModel):
            self._model = model
        elif isinstance(model, str) and model.startswith('arima'):
            self._model = self._get_arima_model(model)
        else:
            raise ValueError('Unknown input for model: %s.'%(model))
        self.data_logger = data_logger
        self.field_to_forecast = field_to_forecast
        self._training_thread_running = False
        self.update_on_copy_of_live_model = update_on_copy_of_live_model
        if self.update_on_copy_of_live_model:
            self._model_for_serving = deepcopy(self._model)
        else:
            self._model_for_serving = self._model
        self.current_time_stamp = None

    def _get_arima_model(self, model_descr):
        """ Obtain an arima model. """
        # pylint: disable=import-outside-toplevel
        from cilantro.timeseries.arima import ARIMATSModel
        arima_args = list(model_descr.split('-')[1:])
        if len(arima_args) <= 1:
            model = ARIMATSModel(self.app_id, (1, 1, 1))
        elif len(arima_args) == 3:
            arima_args = [int(elem) for elem in arima_args]
            order = tuple(arima_args)
            model = ARIMATSModel(self.app_id, order)
        elif len(arima_args) == 4:
            arima_args = [int(elem) for elem in arima_args]
            order = tuple(arima_args[:3])
            max_data_length = arima_args[3]
            model = ARIMATSModel(self.app_id, order, max_data_length)
        else:
            raise ValueError('Invalid input argument %s for model_descr.'%(model_descr))
        return model

    def initialise(self):
        """ Initialises learner. """
        self._model.initialise_model()

    def model_update_routine(self):
        """ Run initialise when you are
        """
        new_data, new_time_stamp = \
            self.data_logger.get_data(fields=[self.field_to_forecast],
                                      start_time_stamp=self.current_time_stamp,
                                      end_time_stamp=None)
        logger.debug('Received %d data for timeseries model %s.', len(new_data),
                     self._model.name)
        if len(new_data) != 0:
            # Process if we have new data samples
            self.current_time_stamp = new_time_stamp
            X = [elem[self.field_to_forecast] for elem in new_data]
            if self.update_on_copy_of_live_model:
                self._model.update_model_with_new_data(X)
                model_deep_copy = deepcopy(self._model)
                self._model_for_serving = model_deep_copy
            else:
                raise NotImplementedError('Not implemented this use case yet.')

    # Obtain recommendations -----------------------------------------------------------------
    def forecast(self, num_steps_ahead=1, conf_alpha=0.90):
        """ Obtain a prediction for the timeseries. """
        return self._model_for_serving.forecast(num_steps_ahead, conf_alpha)

