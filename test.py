import os, argparse
from tqdm import tqdm
from collections import OrderedDict

import torch

from option import option
from data import create_dataset, create_dataloader
from model import define_C
from solver import create_solver
from util.metric import MetricMeter

def main():
    parser = argparse.ArgumentParser(
        description='Train Super Resolution Models')
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to options JSON file.')
    opt = option.parse(parser.parse_args().opt)
    
    
    test_loader_dict = OrderedDict()
    for name, dataset_opt in opt['datasets'].items():
        test_set = create_dataset(dataset_opt)
        test_loader_dict[name] = create_dataloader(val_set, dataset_opt)
        print('===> Test Dataset: %s   Number of images: [%d]' %
              (name, len(test_set)))
        
    solver = create_solver(opt)
    
    print('===> Start Test')
    print("==================================================")
    
    result_dict = OrderedDict()
    metric_meter = MetricMeter(class_num=20)
    
    for name, dl in test_loader_dict.items():
        print('===> Testing %s' % name)
        with tqdm(total=len(dl), desc=name, miniters=1) as t:
            for iter, batch in enumerate(dl):
                solver.feed_data(batch)
                iter_loss, heatmap_dict = solver.test()
                metric_meter.add(solver.predict, solver.target)
                t.set_postfix_str("%s: %.4f" % (batch['img_name'], iter_loss['loss_total']))
                t.update()
                
        metric_value = metric_meter.value()
        result_dict[name] = metric_value
        metric_meter.reset()
        
        print(
            "[%s] mAP: %.2f   mAcc: %.4f   wAcc: %.4f"
            % (name, metric_value['mAP'],
               metric_value['mAcc'], metric_value['wAcc']))
        
    with open(os.path.join(opt['path']['results_root'], 'result.json'), 'w') as f:
        json.dump(result_dict, f, indent=4)
    