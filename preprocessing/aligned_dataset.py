import os.path
import random
import torchvision.transforms as transforms
import torch
import torch.utils.data as data
from PIL import Image
from .utils import get_images_list


class AlignedDataset(data.Dataset):
    def __init__(self, dataroot, is_training=True, should_crop=False):
        super(AlignedDataset,self).__init__()
        self.scale_size = 286
        self.crop_size = 256 # Also known as fine size, for the smaller version
        self.should_crop = should_crop
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

        if self.should_crop:
            w_offset = int(round(width - self.crop_size) / 2.0)
            h_offset = int(round(height - self.crop_size) / 2.0)
        else:
            w_offset = random.randint(0, max(0,width - self.crop_size - 1))
            h_offset = random.randint(0, max(0,height - self.crop_size - 1))

        image_height = h_offset + self.crop_size
        image_width = w_offset + self.crop_size

        A_image = AB[:, h_offset:image_height, w_offset:image_width]
        B_image = AB[:, h_offset:image_height, width + w_offset: width + image_width]

        A_image = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(A_image)
        B_image = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))(B_image)

        #TODO: Finish up https://github.com/junyanz/BicycleGAN/blob/master/data/aligned_dataset.py (line 43-62)
        return {'A': A_image, 'B': B_image}
