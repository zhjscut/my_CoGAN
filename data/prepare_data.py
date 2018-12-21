import torch
import torchvision.transforms as transforms
import torchvision.utils as utils
import os
from PIL import Image
import numpy as np

from utils import (
    imread,
    imwrite,
    )

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

preprocess = transforms.Compose([
    transforms.ToTensor(),
])

def binarization(image):
    image[image > 0.5] = 1
    image[image <= 0.5] = 0
    return image

def default_loader(path):
    img = imread(path, as_gray=True)
    img_tensor = preprocess(img)
    img_tensor = binarization(img_tensor) # 0-1 two value float tensor
    return img_tensor


class my_dataset(torch.utils.data.Dataset):
    def __init__(self, directory, loader=default_loader):
        self.directory = directory
        self.paths = os.listdir(self.directory)
        try:
            paths.remove('img.txt')
        except:
            pass
        self.paths = self.paths[:64]
        self.loader = loader
    
    def __getitem__(self, index):
        path = self.paths[index].split(' ')[0]
        # label = self.paths[index].split(' ')[1][:-1] # 尚不知是要str还是int，后面看需要调整
        label = 0 # label is not needed here, so give it an arbitrary constant value
        image = self.loader(os.path.join(self.directory, path))
        return image, label
    
    def __len__(self):
        return len(self.paths)


def generate_dataloader(args):
    train_dataset_source = my_dataset(args.data_path_source)
    train_dataset_target = my_dataset(args.data_path_target)
    train_loader_source = torch.utils.data.DataLoader(dataset=train_dataset_source, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True, shuffle=True)
    train_loader_target = torch.utils.data.DataLoader(dataset=train_dataset_target, num_workers=args.num_workers, batch_size=args.batch_size, pin_memory=True, shuffle=True)

    return train_loader_source, train_loader_target



