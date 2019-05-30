import pickle
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from .block import GraphConvolution, SoftProposal
from util.util import gen_A, gen_adj


class densenet161(nn.Module):
    def __init__(self, opt):
        super(densenet161, self).__init__()
        self.model_ft = models.densenet161(pretrained=True)
        num_ftrs = self.model_ft.classifier.in_features
        # no Sigmoid
        self.model_ft.classifier = nn.Sequential(nn.Linear(num_ftrs, opt['num_class']))
        if opt['sp']:
            self.SP = SoftProposal(opt['sp']['N'], opt['sp']['max_iter'], opt['sp']['err_th'])
        else:
            self.SP = None

    def forward(self, x, need_feature=False):
        features = self.model_ft.features(x)
        if self.SP:
            features = self.SP(features)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.model_ft.classifier(out)
        
        if need_feature:
            return out, features
        else:
            return out
        
    def get_linear_weight(self):
        return list(self.model_ft.classifier.parameters())[0].data.cpu().numpy()


class densenet201(nn.Module):
    def __init__(self, opt):
        super(densenet201, self).__init__()
        self.model_ft = models.densenet201(pretrained=True)
        num_ftrs = self.model_ft.classifier.in_features
        # no Sigmoid
        self.model_ft.classifier = nn.Sequential(nn.Linear(num_ftrs, opt['num_class']))

    def forward(self, x, need_feature=False):
        features = self.model_ft.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1)).view(features.size(0), -1)
        out = self.model_ft.classifier(out)
        
        if need_feature:
            return out, features
        else:
            return out

    def get_linear_weight(self):
        return list(self.model_ft.classifier.parameters())[0].data.cpu().numpy()

class vgg19(nn.Module):
    def __init__(self, opt):
        super(vgg19, self).__init__()
        model_ft = models.vgg19_bn(pretrained=True)
        self.features = nn.Sequential(*list(model_ft.features.children())[:-1])
        # no Sigmoid
        self.classifier = nn.Sequential(nn.Linear(512, opt['num_class']))

    def forward(self, x, need_feature=False):
        features = self.features(x)
        out = F.adaptive_avg_pool2d(features, (1, 1)).view(features.size(0), -1)
        out = self.classifier(out)
        if need_feature:
            return out, features
        else:
            return out

    def get_linear_weight(self):
        return list(self.classifier.parameters())[0].data.cpu().numpy()

########
# GCN
########

class densenet161GCN(nn.Module):
    def __init__(self, opt):
        super(densenet161GCN, self).__init__()
        self.model_ft = models.densenet161(pretrained=True)
        self.num_ftrs = self.model_ft.classifier.in_features
        self.num_class = opt['num_class']

        self.pooling = nn.AdaptiveMaxPool2d(1)
        self.gc1 = GraphConvolution(opt['in_channel'],1024)
        self.gc2 = GraphConvolution(1024,2208)
        self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A(self.num_class, opt['t'], opt['adj_file'])
        self.A = nn.Parameter(torch.from_numpy(_adj).float().to(torch.device('cuda')))
        
        with open(opt['inp_name'],'rb') as f:
            inp = pickle.load(f)
        inp = inp.astype(np.float32)
        self.inp = torch.from_numpy(inp).cuda()


    def forward(self, x):

        features = self.model_ft.features(x)
        features = F.relu(features, inplace=True)
        adj = gen_adj(self.A).detach()

        features = self.pooling(features)
        features = features.view(features.size(0),features.size(1))
        
        #inp = inp[0]

        #import pdb;pdb.set_trace()
        x = self.gc1(self.inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)
        
        x = x.transpose(0,1)
        x = torch.matmul(features,x)
        return x


class resnet101GCN(nn.Module):
    def __init__(self, opt):
        super(resnet101GCN, self).__init__()
        model_ft = models.resnet101(pretrained=True)
        self.feature = nn.Sequential(
            model_ft.conv1,
            model_ft.bn1,
            model_ft.relu,
            model_ft.maxpool,
            model_ft.layer1,
            model_ft.layer2,
            model_ft.layer3,
            model_ft.layer4,
        )

        self.num_class = opt['num_class']

        self.pooling = nn.AdaptiveMaxPool2d(1)
        self.gc1 = GraphConvolution(opt['in_channel'],1024)
        self.gc2 = GraphConvolution(1024,2048)
        self.relu = nn.LeakyReLU(0.2)

        _adj = gen_A(self.num_class, opt['t'], opt['adj_file'])
        self.A = nn.Parameter(torch.from_numpy(_adj).float().to(torch.device('cuda')))
        
        with open(opt['inp_name'],'rb') as f:
            inp = pickle.load(f)
        inp = inp.astype(np.float32)
        self.inp = torch.from_numpy(inp).cuda()

    def forward(self, x):

        features = self.feature(x)
        features = F.relu(features)
        adj = gen_adj(self.A).detach()
        features = self.pooling(features)
        features = features.view(features.size(0),features.size(1))

        #import pdb;pdb.set_trace()
        x = self.gc1(self.inp, adj)
        x = self.relu(x)
        x = self.gc2(x, adj)
        
        x = x.transpose(0,1)
        x = torch.matmul(features,x)
        return x
