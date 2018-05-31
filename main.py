import torch
import numpy as np
import matplotlib.pyplot as plt
import os
from preprocessing.aligned_dataset import AlignedDataset
from torchvision.utils import make_grid
from model.bicycle_gan import BicycleGAN

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.waitforbuttonpress()

train_dataset = AlignedDataset(dataroot="./data/datasets/maps")

test_img = train_dataset[1]
A = test_img['A']
B = test_img['B']

# show(make_grid([A,B], padding=100))

options = {
    'AtoB': True,
    'training': True,
    'batch_size': 8,
    'lambda_GAN': 1.0, #Weight for the weight on D loss. D(G(A, E(B)))
    'lambda_GAN2': 1.0, # weight on D2 loss, D(G(A, random_z))
    'no_encoding': False,
    'num_gen_filters': 64, #num of gen filters in first conv layer
    'activation': 'relu', # relu | lrelu | elu
    'where_add_z': 'all', # all | middle | input
    'device': torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu'),
    'dir': os.path.join(os.sep, "data","datasets","maps"),
    'input_channel': 3,
    'output_channels': 3,
    'num_latent_vec': 8
}

if __name__ == "__main__":
    model = BicycleGAN()
    model.initialise( options )

