import torch
from preprocessing.aligned_dataset import AlignedDataset

train_dataset = AlignedDataset(dataroot="./data/datasets/maps")

test_img = train_dataset[1]
print(test_img.size())
