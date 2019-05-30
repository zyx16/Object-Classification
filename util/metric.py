import torch
from torchnet.meter import mAPMeter

import numpy as np

class MetricMeter(object):
    def __init__(self, class_num=20):
        self.mAPMeter = mAPMeter()
        self.class_num = class_num
        self.true_array = np.zeros((1, class_num))
        self.total = 0
        self.cnt_array = np.zeros((1, class_num)) # count occurrence of labels
        
    def add(self, out, target_t):
        '''
        out: B*K
        target_t: B*K
        '''
        self.mAPMeter.add(out, target_t)
        if isinstance(out, torch.cuda.FloatTensor):
            output = out.cpu().numpy() > 0.5
            target = target_t.cpu().numpy()
        elif isinstance(out, torch.FloatTensor):
            output = out.numpy() > 0.5
            target = target_t.numpy()
        else: # numpy array
            output = out > 0.5
        result = (target.astype(bool) == output).astype(int) # if true result=1 else res=0
        self.true_array += np.sum(result, axis=0)
        self.cnt_array += np.sum(target, axis=0)
        self.total += out.shape[0]
        
    def value(self):
        mAcc = np.mean(self.true_array.astype(float) / self.total)
        wAcc = np.sum(self.true_array.astype(float) / self.total * self.cnt_array / np.sum(self.cnt_array))
        mAP = self.mAPMeter.value().item()
        return {'mAcc': mAcc, 'wAcc': wAcc, 'mAP': mAP}
    
    def reset(self):
        self.mAPMeter.reset()
        self.true_array = np.zeros((1, self.class_num))
        self.total = 0
        self.cnt_array = np.zeros((1, self.class_num))
