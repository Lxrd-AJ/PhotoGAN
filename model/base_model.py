import os
import torch 
from collections import OrderedDict
from . import cnn_utils

class BaseModel():
    def name(self):
        return "BaseModel"

    def initialise(self, device, is_training, dir):
        self.device = device