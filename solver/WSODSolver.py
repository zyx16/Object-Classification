import torch
import torch.nn as nn
from torch import optim

import os
import cv2
from collections import OrderedDict
import numpy as np

from .BaseSolver import BaseSolver
from model import define_C
from util.util import find_bbox

class WSODSolver(BaseSolver):
    def __init__(self, opt):
        super(WSODSolver, self).__init__(opt)
        assert not self.is_train # only use this when test
        self.train_opt = opt['solver']
        self.input_img = self.Tensor()
        self.target = self.Tensor()

        self.netC = define_C(opt)
        self.print_network()
        self.load()
        self.weight = self.get_linear_weight()
        self.input_size = opt['datasets'][list(opt['datasets'].keys())[0]]['input_size']
        print('===> Solver Initialized : [%s] || Use GPU : [%s]' % (self.__class__.__name__, self.use_gpu))


    def feed_data(self, sample, need_bbox=False):
        input_img = sample['img']
        target = sample['label']
        self.input_img.resize_(input_img.size()).copy_(input_img)
        self.target.resize_(target.size()).copy_(target)

    def test(self):
        self.netC.eval()
        with torch.no_grad():
            output, feature = self.netC(self.input_img, need_feature=True)
            output = nn.Sigmoid()(output)
            self.predict = output
            self.feature = feature

    def get_CAMs(self):
        feature = self.feature.cpu().numpy()
        b, c, h, w = feature.shape
        CAM_list = []
        for i in range(b):
            res = np.dot(self.weight, np.reshape(feature[i], (c, h * w)))
            res = np.reshape(res, (-1, h, w))
            res = np.transpose(res, (1, 2, 0))
            CAM_list.append(cv2.resize(res, (self.input_size, self.input_size), interpolation=cv2.INTER_LINEAR))
        self.CAM_list = CAM_list

    def detect_box(self):
        result_dict = {}
        for box_th in self.train_opt['bbox_th']:
            result_list = []
            for i, CAM in enumerate(self.CAM_list): # i: batch num
                predict_labels = []
                bbox_dict = OrderedDict()
                # get labels
                if torch.sum(self.predict[i].cpu() > self.train_opt['pred_th'])  == 0: # no predict
                    predict_labels.append(torch.argmax(self.predict[i]))
                else:
                    for l, p in enumerate(self.predict[i].cpu()):
                        if p > self.train_opt['pred_th']:
                            predict_labels.append(l)
                # get bbox
                for l in predict_labels:
                    # [bbox1, bbox2, ..., bboxN, pred]
                    box, _ = find_bbox(CAM[:, :, l], box_th)
                    if len(box) <= 0:
                        continue
                    bbox_dict[l] = box
                    bbox_dict[l].append(self.predict[i].cpu()[l].item())
                    
                result_list.append(bbox_dict)
            result_dict[box_th] = result_list
        return result_dict
        
    def load(self):
        """
        load
        """

        model_path = self.opt['solver']['pretrained_path']
        if model_path is None:
            raise ValueError(
                "[Error] The 'pretrained_path' does not declarate in *.json"
            )

        print('===> Loading classifier model from [%s]...' % model_path)
        load_func = self.netC.module.load_state_dict if isinstance(self.netC, nn.DataParallel) \
                else self.netC.load_state_dict

        checkpoint = torch.load(model_path)
        if isinstance(checkpoint, dict):
            state_dict = checkpoint['netC']
        elif isinstance(checkpoint, OrderedDict):
            state_dict = checkpoint
        else:
            raise ValueError('Can\'t load from %s' % (model_path))
        load_func(state_dict)

    def get_linear_weight(self):
        weight = self.netC.module.get_linear_weight() if isinstance(self.netC, nn.DataParallel) \
                else self.netC.get_linear_weight()
        return weight

    def print_network(self):
        """
        print network summary including module and number of parameters
        """
        s, n = self.get_network_description(self.netC)
        if isinstance(self.netC, nn.DataParallel):
            net_struc_str = '{} - {}'.format(
                self.netC.__class__.__name__,
                self.netC.module.__class__.__name__)
        else:
            net_struc_str = '{}'.format(self.netC.__class__.__name__)

        print("==================================================")
        print("===> Network Summary\n")
        net_lines = []
        line = s + '\n'
        # print(line)
        net_lines.append(line)
        line = 'Network structure: [{}], with parameters: [{:,d}]'.format(
            net_struc_str, n)
        print(line)
        net_lines.append(line)

        if self.is_train:
            with open(os.path.join(self.exp_root, 'network_summary.txt'),
                      'w') as f:
                f.writelines(net_lines)

        print("==================================================")
