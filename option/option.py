import os
import os.path as osp
from collections import OrderedDict
import json

from util import util

def parse(opt_path, is_train=True):
    # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    # datasets
    for phase, dataset in opt['datasets'].items():
        dataset['phase'] = phase

    # path
    if is_train:
        experiments_root = os.path.join(opt['path']['root'], 'experiments', opt['name'])
        opt['path']['exp_root'] = experiments_root
        opt['path']['checkpoint_dir'] = os.path.join(experiments_root, 'checkpoint')
        if opt['use_tb_logger']:
            opt['path']['tb_logger'] = os.path.join(opt['path']['root'], 'tb_logger', opt['name'])
            
        # create folders
        if opt['solver']['pretrain'] != 'resume' and not 'debug' in opt['name']:
            util.mkdir_and_rename(experiments_root)
            if opt['use_tb_logger']:
                util.mkdir_and_rename(opt['path']['tb_logger'])
            util.mkdirs((path for key, path in opt['path'].items() if key != 'exp_root' and key != 'tb_logger_root'))
            
    else:  # test
        results_root = os.path.join(opt['path']['root'], 'results', opt['name'])
        opt['path']['results_root'] = results_root
        opt['path']['log'] = results_root
        if not 'debug' in opt['name']:
            util.mkdir_and_rename(results_root)


    # network

    # export CUDA_VISIBLE_DEVICES
    gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)

    return dict_to_nonedict(opt)

def check_resume(opt):
    '''Check resume states and pretrain_model paths'''
    if opt['path']['resume_state']:
        if opt['path']['pretrain_model_G'] or opt['path']['pretrain_model_D']:
            print('===> Warning pretrain_model path will be ignored when resuming training.')

        state_idx = osp.basename(opt['path']['resume_state']).split('.')[0]
        opt['path']['pretrain_model_G'] = osp.join(opt['path']['models'],
                                                   '{}_G.pth'.format(state_idx))
        print('===> Info Set [pretrain_model_G] to ' + opt['path']['pretrain_model_G'])
        if 'gan' in opt['model']:
            opt['path']['pretrain_model_D'] = osp.join(opt['path']['models'],
                                                       '{}_D.pth'.format(state_idx))
            print('===> Info Set [pretrain_model_D] to ' + opt['path']['pretrain_model_D'])




class NoneDict(dict):
    def __missing__(self, key):
        return None


# convert to NoneDict, which return None for missing key.
def dict_to_nonedict(opt):
    if isinstance(opt, dict):
        new_opt = dict()
        for key, sub_opt in opt.items():
            new_opt[key] = dict_to_nonedict(sub_opt)
        return NoneDict(**new_opt)
    elif isinstance(opt, list):
        return [dict_to_nonedict(sub_opt) for sub_opt in opt]
    else:
        return opt
