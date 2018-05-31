
import torch
from .base_model import BaseModel
from . import cnn_utils


class BicycleGAN(BaseModel):
    def name(self):
        return 'BicycleGANModel'

    def initialise(self, options):
        self.is_training = options['training']
        self.batch_size = options['batch_size']
        if self.is_training:
            assert self.batch_size % 2 == 0
        # BaseModel.initialise( options['device'], self.is_training, options['dir'] )
        self.loss_names = ['Gen_GAN', 'Gen_GAN_2', 'D_GAN', 'D_GAN_2', 'Gen_L1', 'z_L1', 'kL' ]
        self.visual_names = ['real_A_encoded', 'real_B_encoded', 'fake_B_random', 'fake_B_encoded']

        # Parameters for the network
        self.model_names = ['G']
        self.netGen = cnn_utils.define_Gen( 3,3,options['num_latent_vec'],64, norm='instance' )
