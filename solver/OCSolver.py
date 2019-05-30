import torch
import torch.nn as nn
from torch import optim

import os
import pandas as pd
from collections import OrderedDict

from .BaseSolver import BaseSolver
from model import define_C

class OCSolver(BaseSolver):
    def __init__(self, opt):
        super(OCSolver, self).__init__(opt)
        self.train_opt = opt['solver']
        self.input_img = self.Tensor()
        self.target = self.Tensor()

        self.records = {'train_loss_c': [],
                        'train_loss_total': [],
                        'val_loss_c': [],
                        'val_loss_total': [],
                        'mAP': [],
                        'mAcc': [],
                        'wAcc': [],
                        'lr': []
                        }

        self.netC = define_C(opt)
        self.print_network()

        if self.is_train:
            self.netC.train()

            loss_type = self.train_opt['loss_type']
            if loss_type == 'BCE':
                self.criterion_C = nn.BCELoss()
            elif loss_type == 'BCEWithLogits':
                self.criterion_C = nn.BCEWithLogitsLoss(pos_weight=self.train_opt['pos_weight'])
            else:
                raise NotImplementedError('Loss type %s not implemented!' % (loss_type))

            if self.use_gpu:
                self.criterion_C = self.criterion_C.cuda()

            weight_decay = self.train_opt['weight_decay'] if self.train_opt['weight_decay'] else 0
            optim_type = self.train_opt['type'].upper()
            if optim_type == "ADAM":
                self.optimizer_C = optim.Adam(self.netC.parameters(),
                        lr=self.train_opt['learning_rate'], weight_decay=weight_decay)
            elif optim_type == "SGD":
                self.optimizer_C = optim.SGD(self.netC.parameters(),
                        lr=self.train_opt['learning_rate'], weight_decay=weight_decay)
            else:
                raise NotImplementedError('Optimizer type %s not implemented!' % (optim_type))

            lr_scheme_type = self.train_opt['lr_scheme'].lower()
            if lr_scheme_type == 'multisteplr':
                self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer_C,
                        self.train_opt['lr_steps'],
                        self.train_opt['lr_gamma'])
            else:
                raise NotImplementedError('Learning rate scheduler type %s not implemented!' % (lr_scheme_type))

        self.load()
        print('===> Solver Initialized : [%s] || Use GPU : [%s]' % (self.__class__.__name__, self.use_gpu))

        if self.is_train:
            print("optimizer_C: ", self.optimizer_C)
            print("lr_scheduler milestones: %s   gamma: %f"%(self.scheduler.milestones, self.scheduler.gamma))

    def feed_data(self, sample, need_bbox=False):
        input_img = sample['img']
        target = sample['label']
        self.input_img.resize_(input_img.size()).copy_(input_img)
        self.target.resize_(target.size()).copy_(target)

    def train_step(self):
        self.netC.train()
        self.optimizer_C.zero_grad()

        output = self.netC(self.input_img)
        if self.train_opt['loss_type'] != 'BCEWithLogits':
            output = nn.Sigmoid()(output)
        loss = self.criterion_C(output, self.target)

        loss.backward()
        self.optimizer_C.step()
        self.netC.eval()

        return {'loss_c': loss.item(), 'loss_total': loss.item()}

    def test(self):
        self.netC.eval()
        with torch.no_grad():
            output = self.netC(self.input_img)
            if self.is_train:
                # no Sigmoid in last layer
                if self.train_opt['loss_type'] != 'BCEWithLogits':
                    output = nn.Sigmoid()(output)
                loss = self.criterion_C(output, self.target)
                # output probability
                if self.train_opt['loss_type'] == 'BCEWithLogits':
                    output = nn.Sigmoid()(output)
            else:
                output = nn.Sigmoid()(output)
            self.predict = output
            self.netC.train()

        if self.is_train:
            return {'loss_c': loss, 'loss_total': loss.item()}

    def save_checkpoint(self, epoch, step, is_best):
        """
        save checkpoint to experimental dir
        """
        filename = os.path.join(self.checkpoint_dir, 'last_ckp.pth')
        print('===> Saving last checkpoint to [%s] ...]' % filename)
        ckp = {
            'epoch': epoch,
            'step': step,
            'netC': self.netC.module.state_dict() if isinstance(self.netC, nn.DataParallel) else self.netC.state_dict(),
            'optimizer_C': self.optimizer_C.state_dict(),
            'best_mAP': self.best_mAP,
            'best_epoch': self.best_epoch,
            'records': self.records
        }
        torch.save(ckp, filename)

        if is_best:
            print('===> Saving best checkpoint to [%s] ...]' %
                  filename.replace('last_ckp', 'best_ckp'))
            torch.save(ckp, filename.replace('last_ckp', 'best_ckp'))

        if epoch % self.train_opt['save_step'] == 0:
            print('===> Saving checkpoint [%d] to [%s] ...]' %
                  (epoch,
                   filename.replace('last_ckp', 'epoch_%d_ckp' % epoch)))

            torch.save(
                ckp, filename.replace('last_ckp', 'epoch_%d_ckp' % epoch))

    def load(self):
        """
        load or initialize network
        """
        if (self.is_train
                and self.opt['solver']['pretrain']) or not self.is_train:
            model_path = self.opt['solver']['pretrained_path']
            if model_path is None:
                raise ValueError(
                    "[Error] The 'pretrained_path' does not declarate in *.json"
                )

            print('===> Loading classifier model from [%s]...' % model_path)
            load_func = self.netC.module.load_state_dict if isinstance(self.netC, nn.DataParallel) \
                    else self.netC.load_state_dict

            if self.is_train:
                checkpoint = torch.load(model_path)
                load_func(checkpoint['netC'])

                # resume state
                if self.opt['solver']['pretrain'] == 'resume':
                    print('===> Resuming state')
                    self.cur_epoch = checkpoint['epoch'] + 1
                    self.cur_step = checkpoint['step'] + 1
                    self.optimizer_C.load_state_dict(checkpoint['optimizer_C'])
                    self.best_mAP = checkpoint['best_mAP']
                    self.best_epoch = checkpoint['best_epoch']
                    self.records = checkpoint['records']
            else:
                checkpoint = torch.load(model_path)
                if isinstance(checkpoint, dict):
                    state_dict = checkpoint['netC']
                elif isinstance(checkpoint, OrderedDict):
                    state_dict = checkpoint
                else:
                    raise ValueError('Can\'t load from %s' % (model_path))
                load_func(state_dict)

    def get_current_learning_rate(self):
        return self.optimizer_C.param_groups[0]['lr']

    def update_learning_rate(self, epoch):
        self.scheduler.step(epoch)

    def get_current_log(self):
        log = OrderedDict()
        log['step'] = self.cur_step
        log['epoch'] = self.cur_epoch
        log['best_mAP'] = self.best_mAP
        log['best_epoch'] = self.best_epoch
        log['records'] = self.records
        return log

    def set_current_log(self, log):
        self.cur_step = log['step']
        self.cur_epoch = log['epoch']
        self.best_mAP = log['best_mAP']
        self.best_epoch = log['best_epoch']
        self.records = log['records']

    def save_current_log(self):
        data_frame = pd.DataFrame(
            data=self.records,
            index=range(1, self.cur_epoch + 1))
        data_frame.to_csv(
            os.path.join(self.exp_root, 'train_records.csv'),
            index_label='epoch')

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
