import torch
import torch.nn as nn
import torchvision
import torchvision.models as models
import os
import cv2
import numpy as np


global features_blobs
def hook_feature(module, input, output):
    global features_blobs
    features_blobs=output.data.cpu().numpy()

class densenet161(nn.Module):
    def __init__(self, opt):
        super(densenet161, self).__init__()
        self.model_ft = models.densenet161(pretrained=True)
        num_ftrs = self.model_ft.classifier.in_features
        # no Sigmoid
        self.model_ft.classifier = nn.Sequential(nn.Linear(num_ftrs, opt['num_class']))
        finalconv_name = 'features'
        self.model_ft._modules.get(finalconv_name).register_forward_hook(hook_feature)
        
    def forward(self, x, opt_heatmap):
        out = self.model_ft(x)
        if opt_heatmap:
            bz, nc, width, height = x.shape
            class_idx = range(20)
            params = list(self.model_ft.parameters())
            weight_softmax = np.squeeze(params[-2].data.cpu().numpy())
            heatmap_dict = dict()
            for idx in class_idx:
                heatmap = returnCAM(features_blobs, weight_softmax, idx)
                heatmap_dict[idx] = heatmap
            return out, heatmap_dict
        else:
            return out

def returnCAM(feature_conv, weight_softmax, idx, input_size=224):
    size_upsample = (input_size, input_size)
    bz, nc, h, w = feature_conv.shape
    output_cam = []
    for b in range(bz):
        cam = weight_softmax[idx].dot(feature_conv[b].reshape((nc, h*w)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)
        output_cam.append(cv2.resize(cam_img, size_upsample))
    return output_cam

class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'

