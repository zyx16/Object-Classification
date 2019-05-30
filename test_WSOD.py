import os, argparse, json
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
        description='Test Super Resolution Models')
    parser.add_argument(
        '-opt', type=str, required=True, help='Path to options JSON file.')
    opt = option.parse(parser.parse_args().opt, is_train=False)
    
    
    test_loader_dict = OrderedDict()
    for name, dataset_opt in opt['datasets'].items():
        test_set = create_dataset(dataset_opt)
        test_loader_dict[name] = create_dataloader(test_set, dataset_opt)
        print('===> Test Dataset: %s   Number of images: [%d]' %
              (name, len(test_set)))
        
    solver = create_solver(opt)
    
    print('===> Start Test')
    print("==================================================")
    
    result_dict = OrderedDict()
    metric_meter = MetricMeter(class_num=20)
    
    for name, dl in test_loader_dict.items():
        print('===> Testing %s' % name)
        res_str = {}
        for k in opt['solver']['bbox_th']:
            res_str[k] = []
        with tqdm(total=len(dl), desc=name, miniters=1) as t:
            for iter, batch in enumerate(dl):
                solver.feed_data(batch)
                solver.test()
                metric_meter.add(solver.predict, solver.target)
                solver.get_CAMs()
                res = solver.detect_box()
                # write to str
                for k, v in res.items():
                    for b, d in enumerate(v):
                        img_name = batch['img_name'][b]
                        img_str_list = []
                        for label, pred in d.items():
                            label_str_list = []
                            for bbox in pred[:-1]:
                                out_str = '%d %f ' % (label, pred[-1])
                                x_scale = batch['original_size'][0][b].item() / float(solver.input_size)
                                y_scale = batch['original_size'][1][b].item() / float(solver.input_size)
                                scale_box = [bbox[0] * x_scale, bbox[1] * y_scale, 
                                             (bbox[0] + bbox[2]) * x_scale, (bbox[1] + bbox[3]) * y_scale]
                                out_str += ' '.join([str(int(x)) for x in scale_box])
                                label_str_list.append(out_str)
                            label_str = '\n'.join(label_str_list)
                            img_str_list.append(label_str)
                        img_str = '\n'.join(img_str_list)
                        res_str[k].append((img_name, img_str))
                        
                t.update()
                
        metric_value = metric_meter.value()
        result_dict[name] = metric_value
        metric_meter.reset()
        
        print(
            "[%s] mAP: %.2f   mAcc: %.4f   wAcc: %.4f"
            % (name, metric_value['mAP'],
               metric_value['mAcc'], metric_value['wAcc']))
        # write bbox
        for k, v in res_str.items():
            write_dir = os.path.join(opt['path']['results_root'], '%s_bbox_th%.2f' % (name, k))
            os.mkdir(write_dir)
            for item in v:
                write_path = os.path.join(write_dir, item[0] + '.txt')
                with open(write_path, 'w') as f:
                    f.write(item[1])
            # move results
            os.system('cp %s/*.txt ./mAP/input/detection-results/' % write_dir)
            print('[%s] th: %.3f' % (name, k))
            os.system('python ./mAP/main.py -na -np -q')
            # read mAP
            with open('./mAP/results/results.txt') as f:
                lines = f.read().splitlines()
            for l in lines:
                if l.startswith('mAP = '):
                    break
            mAP = float(l.split(' ')[-1][:-1])
            result_dict[name][k] = mAP
            
        # write json
        with open(os.path.join(opt['path']['results_root'], '%s_result.json' % name), 'w') as f:
            json.dump(result_dict, f, indent=4)
    
if __name__ == '__main__':
        main()
