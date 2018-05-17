import os.path
import random
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
from PIL import Image
from .utils import get_images_list


class AlignedDataset(data.Dataset):
    def __init__(self, dataroot, is_training=True):
        super(AlignedDataset,self).__init__()
        self.scale_size = 286
        self.crop_size = 256
        self.initialise(dataroot, is_training)
    
    def name(self):
        return "AlignedDataset"

    def initialise(self, dataroot, is_training=True):
        self.data_dir = os.path.join(dataroot, "train" if is_training else "val")
        self.AB_paths = sorted(get_images_list(self.data_dir))
        
    def __getitem__(self, index):
        AB_path = self.AB_paths[index]
        AB = Image.open(AB_path).convert('RGB')
        AB = AB.resize((self.scale_size * 2, self.scale_size), Image.BICUBIC)
        AB = transforms.ToTensor()(AB)

        total_width = AB.size(2)
        width = int(total_width/2)
        height = AB.size(1)

        #TODO: Continue center crop
        return AB
