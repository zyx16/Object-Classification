import os
from datetime import datetime
import cv2
import matplotlib.pyplot as plt
from data.dataset import UnNormalize
from PIL import Image
from torchvision import transforms
import torch
import numpy as np
import math
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
        
def gen_A(num_classes, t=0.4, adj_file='data/voc_adj.pkl'):
    import pickle
    result = pickle.load(open(adj_file, 'rb'))
    _adj = result['adj']
    _nums = result['nums']
    _nums = _nums[:, np.newaxis]
    _adj = _adj / _nums
    _adj[_adj < t] = 0
    _adj[_adj >= t] = 1
    _adj = _adj * 0.25 / (_adj.sum(0) + 1e-6)
    _adj = _adj + np.identity(num_classes, np.int)
    return _adj

def gen_adj(A):
    D = torch.pow(A.sum(1).float(), -0.5)
    D = torch.diag(D)
    adj = torch.matmul(torch.matmul(A, D).t(), D)
    return adj


