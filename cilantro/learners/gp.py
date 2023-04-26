"""
    A GP Learning model. This is just a LearningModel wrapper for Dragonfly.
    -- kirthevasank
    -- romilbhardwaj
"""

from argparse import Namespace
import logging
from dragonfly.gp.euclidean_gp import EuclideanGPFitter as EGPFitter
# local
from cilantro.learners.base_learner import LearningModel


logger = logging.getLogger(__name__)

LOAD_NORMALISER = 10000


class GP(LearningModel):
    """ A learning model. """
    # pylint: disable=abstract-method

    def __init__(self, name, app_client_key, alloc_leaf_order, options=None):
        """ Constructor.
            - alloc_leaf_order is the order in which allocations for each leaf will be stored in the
                               model.
            - app_client_key is the key to search for in the Rewards and Loads dictionaries for the
                               application's reward and leaf.
        """
        super().__init__(name, int_lb=None, int_ub=None)
        self.alloc_leaf_order = alloc_leaf_order
        self.num_dims = len(alloc_leaf_order)
        self.app_client_key = app_client_key
        self.all_inputs = []
        self.all_labels = []
        self.total_data = 0
        self.curr_model = None
        if options is None:
            self.gp_options = Namespace(
                kernel_type='esp',
                esp_order=5,
                esp_kernel_type='matern',
                esp_matern_nu=2.5
                )
        else:
            self.gp_options = options

    def _initialise_model_child(self):
        """ Initialises model. """
        pass

    def update_model_with_new_data(self, Allocs, Rewards, Loads, Sigmas, Event_times):
        """ Updates the model with new data. """
        num_new_data = 0
        for alloc, reward, load in zip(Allocs, Rewards, Loads):
            new_input = [load/LOAD_NORMALISER] + [alloc[leaf] for leaf in self.alloc_leaf_order]
            self.all_inputs.append(new_input)
            self.all_labels.append(reward)
            num_new_data += 1
#             logger.info('data: ' + str(new_input) + ',   reward: ' + str(reward))
        if num_new_data == 0: # Return if there is no new data
            return
        self.total_data += num_new_data
        logger.info(f'Updated model with {num_new_data} data')
        # Create a GP Fitter and fit the GP
        gp_fitter = EGPFitter(self.all_inputs, self.all_labels)
        _, fitted_gp, _ = gp_fitter.fit_gp()
        self.curr_model = fitted_gp

    def get_mean_pred_and_std_for_alloc_load(self, alloc, load):
        """ Returns the prediction for allocation and load. """
        dat_in = [[load/LOAD_NORMALISER] + [alloc[leaf] for leaf in self.alloc_leaf_order]]
        mean_pred, std = self.curr_model.eval(dat_in, uncert_form='std')
        ret = (mean_pred[0], std[0])
#         logger.info('dat_in: %s', str(dat_in))
#         logger.info('mean_pred, std, ret: %s, %s, %s', mean_pred, std, ret)
        return ret

