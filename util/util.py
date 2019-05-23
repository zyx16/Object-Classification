import os
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
from data.dataset import UnNormalize
from PIL import Image
from torchvision import transforms
import numpy as np
def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('[Warning] Path [%s] already exists. Rename it to [%s]' % (path, new_name))
        os.rename(path, new_name)
    os.makedirs(path)

def plt_heatmap(img_name, img, heatmap_dict):
    detransform = transforms.Compose([
        UnNormalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
        transforms.ToPILImage()])
    img = detransform(img)
    # fig = plt.figure(figsize=(8,10))
    for idx in range(20):
        heatmap = np.array(heatmap_dict[idx])
        heatmap = heatmap.reshape((224,224))
        plt.subplot(4,5,idx+1)        
        plt.title(str(idx))
        plt.imshow(img, alpha=0.8)
        plt.imshow(heatmap, alpha=0.3, cmap='hot')
    plt.savefig('/home/stevetod/jzy/projects/Object-Classification/result/figs/heatmap'+img_name+'.jpg')

        



