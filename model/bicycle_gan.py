
import torch
from .base_model import BaseModel
from . import cnn_utils


class BicycleGANModel(BaseModel):
    def name(self):
        return 'BicycleGANModel'

    def initialise(self, options):
        self.is_training = options['training']
        BaseModel.initialise( options['device'], self.is_training, options['dir'] )
        