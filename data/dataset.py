import os
from PIL import Image

import torch
import torch.nn
from torch.utils.data import Dataset
from torchvision import transforms


class PascalVOCDataset(Dataset):
    def __init__(self, opt):
        self.opt = opt
        self.info_list = info_reader(opt['info_path'])
        self.root_path = opt['root_path']

        # transform
        transform_list = []
        if opt['phase'] == 'train':
            transform_list.append(transforms.RandomResizedCrop(opt['input_size'], scale=opt['scale']))
            if opt['use_flip']:
                transform_list.append(transforms.RandomHorizontalFlip())
        elif opt['phase'] == 'val' or opt['phase'] == 'test':
            transform_list.extend([
                transforms.Resize((opt['input_size'], opt['input_size']))
                ])
        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.transform = transforms.Compose(transform_list)
        self.detransform = transforms.Compose([
            UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
            transforms.ToPILImage()])

        self.category_num = opt['category_num']
            
    def __getitem__(self, index):
        info = self.info_list[index]
        img = Image.open(get_full_path(self.root_path, info[0])).convert('RGB')
        target = torch.zeros(self.category_num)
        target[info[1]] = 1
        return {'img': self.transform(img), 'label': target, 'img_name': info[0]}
    
    def __len__(self):
        return len(self.info_list)
    
    def translate_label(self, label):
        category_list = ["person", "bird", "cat", "cow", "dog", "horse", "sheep", "aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train", "bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"]
        return_list = [category_list[i] for i in range(self.category_num) if label[i] > 0]
        return return_list
    
    def name(self):
        return 'PascalVOC_' + self.opt['phase'] 

class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor
    
    
def info_reader(info_path):
    with open(info_path, 'r') as f:
        info_lines = f.read().splitlines()
    info_list = []
    for i in info_lines:
        split = i.split(' ')
        info_list.append((split[0], [int(x) for x in split[1:]]))
    return info_list

def get_full_path(root_path, name):
    return os.path.join(root_path, name+'.jpg')
