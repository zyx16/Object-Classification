import torch
import torch.nn as nn
import torchvision


class densenet161(nn.Module):
    def __init__(self, opt):
        super(densenet161, self).__init__()
        self.model_ft = torchvision.models.densenet161(pretrained=True)
        num_ftrs = self.model_ft.classifier.in_features
        # no Sigmoid
        self.model_ft.classifier = nn.Sequential(nn.Linear(num_ftrs, opt['num_class']))

    def forward(self, x):
        return(self.model_ft(x))
