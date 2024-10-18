from abc import abstractmethod
import torch
from PIL import Image
import json
import numpy as np
import torchvision.transforms as transforms
import os
import random
import data.additional_transforms as add_transforms

identity = lambda x: x


class DataManager:
    @abstractmethod
    def get_data_loader(self, data_file, aug):
        pass


class TransformLoader:
    def __init__(self, image_size, normalize_param=dict(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                 jitter_param=dict(Brightness=0.4, Contrast=0.4, Color=0.4)):
        self.image_size = image_size
        self.normalize_param = normalize_param
        self.jitter_param = jitter_param

    def parse_transform(self, transform_type):
        if transform_type == 'ImageJitter':
            method = add_transforms.ImageJitter(self.jitter_param)
            return method
        method = getattr(transforms, transform_type)

        if transform_type == 'RandomResizedCrop':
            return method(self.image_size)
        elif transform_type == 'CenterCrop':
            return method(self.image_size)
        elif transform_type == 'Resize':
            return method([int(self.image_size * 1.15), int(self.image_size * 1.15)])
        elif transform_type == 'Normalize':
            return method(**self.normalize_param)
        else:
            return method()

    def get_composed_transform(self, aug=False):
        if aug:
            transform_list = ['RandomResizedCrop', 'ImageJitter', 'RandomHorizontalFlip', 'ToTensor', 'Normalize']
        else:
            transform_list = ['Resize', 'CenterCrop', 'ToTensor', 'Normalize']

        transform_funcs = [self.parse_transform(x) for x in transform_list]
        transform = transforms.Compose(transform_funcs)
        return transform


class SimpleDataManager(DataManager):  # Full class
    def __init__(self, image_size, batch_size):
        super(SimpleDataManager, self).__init__()
        self.batch_size = batch_size
        self.trans_loader = TransformLoader(image_size)
        self.image_size = image_size

    def get_data_loader(self, data_file, aug):
        transform = self.trans_loader.get_composed_transform(aug)

        dataset = SimpleDataset(data_file, transform)
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True, num_workers=0,
                                                  pin_memory=True)
        cls_num = len(dataset.meta['label_names'])
        return dataset, data_loader, cls_num


class SimpleDataset:
    def __init__(self, data_file, transform, target_transform=identity):
        with open(data_file, 'r') as f:
            self.meta = json.load(f)
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, i):
        image_path = os.path.join(self.meta['image_names'][i])
        img = Image.open(image_path).convert('RGB')

        img = self.transform(img)

        target = self.target_transform(self.meta['image_labels'][i])
        return img, target

    def __len__(self):
        return len(self.meta['image_names'])

# if __name__ == '__main__':
#     base_file = os.path.join(params.data_dir, params.dataset, 'base.json')
#     base_datamgr = SimpleDataManager(224, batch_size=16)
#     base_loader = base_datamgr.get_data_loader(base_file, aug=params.train_aug)
