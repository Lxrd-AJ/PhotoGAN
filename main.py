import torch
import numpy as np
import matplotlib.pyplot as plt
from preprocessing.aligned_dataset import AlignedDataset
from torchvision.utils import make_grid

def show(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1,2,0)), interpolation='nearest')
    plt.waitforbuttonpress()

train_dataset = AlignedDataset(dataroot="./data/datasets/maps")

test_img = train_dataset[1]
A = test_img['A']
B = test_img['B']

show(make_grid([A,B], padding=100))




