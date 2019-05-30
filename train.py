import os, argparse
from tqdm import tqdm

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
    
    # DATAESET
    for phase, dataset_opt in opt['datasets'].items():
        if phase == 'train':
            train_set = create_dataset(dataset_opt)
            train_loader = create_dataloader(train_set, dataset_opt)
            print('===> Train Dataset: %s   Number of images: [%d]' %
                  (train_set.name(), len(train_set)))
            if train_loader is None:
                raise ValueError("[Error] The training data does not exist")
        elif phase == 'val':
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt)
            print('===> Val Dataset: %s   Number of images: [%d]' %
                  (val_set.name(), len(val_set)))
        else:
            raise NotImplementedError('Dataset phase %s is not implemented!' % (phase))

    if opt['solver']['balance_sample']:
        pos_weight = torch.zeros(20).float()
        for info in train_set.info_list:
            pos_weight[info[1]] += 1
        pos_weight = (len(train_set) - pos_weight) / pos_weight
        opt['solver']['pos_weight'] = pos_weight
        print('===> Using sample balance, weights are')
        print(pos_weight)
        
    # SOLVER
    solver = create_solver(opt)
    
    # TB_LOGGER
    if opt['use_tb_logger']:
        from tensorboardX import SummaryWriter
        tb_logger = SummaryWriter(log_dir=opt['path']['tb_logger'])
        print('===> tensorboardX logger created, log to %s' %
              (opt['path']['tb_logger']))

    print('===> Start Train')
    print("==================================================")

    # INITIALIZE LOG
    solver_log = solver.get_current_log()
    NUM_EPOCH = int(opt['solver']['epoch'])
    start_epoch = solver_log['epoch']
    current_step = solver_log['step']
    model_name = opt['network_C']['which_model_C'].upper()
    print("Method: %s || Epoch Range: (%d ~ %d) || Start Step: %d" %
          (model_name, start_epoch, NUM_EPOCH, current_step))

    metric_meter = MetricMeter(class_num=20)
    
    for epoch in range(start_epoch, NUM_EPOCH + 1):
        metric_meter.reset()
        print('\n===> Training Epoch: [%d/%d]...  Learning Rate: %f' %
              (epoch, NUM_EPOCH, solver.get_current_learning_rate()))

        # Initialization
        solver_log['epoch'] = epoch

        # Train model
        train_loss_dict = {}
        val_loss_dict = {}
        for k in solver_log['records'].keys():
            if k.startswith('train'):  # 'train_loss_pixel'
                train_loss_dict[k[6:]] = []
            elif k.startswith('val'):
                val_loss_dict[k[4:]] = []

        with tqdm(
                total=len(train_loader),
                desc='Epoch: [%d/%d]' % (epoch, NUM_EPOCH),
                miniters=1) as t:
            for iter, batch in enumerate(train_loader):
                current_step += 1
                solver.feed_data(batch)
                iter_loss = solver.train_step()
                batch_size = batch['img'].size(0)
                for k, v in iter_loss.items():
                    train_loss_dict[k].append(v * batch_size)
                if opt['use_tb_logger']:
                    if current_step % opt['logger']['print_freq'] == 0:
                        for k, v in iter_loss.items():
                            tb_logger.add_scalar('train_' + k, v, current_step)

                t.set_postfix_str("Batch Loss: %.4f" % iter_loss['loss_total'])
                t.update()

        for k, v in train_loss_dict.items():
            solver_log['records']['train_' + k].append(sum(v) / len(v))
        solver_log['records']['lr'].append(solver.get_current_learning_rate())

        print(
            '\nEpoch: [%d/%d]   Avg Train Loss: %.6f' %
            (epoch, NUM_EPOCH, solver_log['records']['train_loss_total'][-1]))

        
        print('===> Validating...', )
        for iter, batch in enumerate(val_loader):
            solver.feed_data(batch)
            iter_loss = solver.test()
            for k, v in iter_loss.items():
                val_loss_dict[k].append(v)
            metric_meter.add(solver.predict, solver.target)

        for k, v in val_loss_dict.items():
            solver_log['records']['val_' + k].append(sum(v) / len(v))
        metric_value = metric_meter.value()
        for k, v in metric_value.items():
            solver_log['records'][k].append(v)

        if opt['use_tb_logger']:
            for k, v in iter_loss.items():
                tb_logger.add_scalar('val_' + k, v, current_step)
            for k, v in metric_value.items():
                tb_logger.add_scalar('val_' + k, v, current_step)

        # record the best epoch
        epoch_is_best = False
        if solver_log['best_mAP'] < metric_value['mAP']:
            solver_log['best_mAP'] = metric_value['mAP']
            epoch_is_best = True
            solver_log['best_epoch'] = epoch

        print(
            "[%s] mAP: %.2f   mAcc: %.4f   wAcc: %.4f Loss: %.6f   Best mAP: %.2f in Epoch: [%d]"
            % (val_set.name(), metric_value['mAP'],
               metric_value['mAcc'], metric_value['wAcc'],
               solver_log['records']['val_loss_total'][-1],
               solver_log['best_mAP'], solver_log['best_epoch']))

        solver.set_current_log(solver_log)
        solver.save_checkpoint(epoch, current_step, epoch_is_best)
        solver.save_current_log()

        # update lr
        solver.update_learning_rate(epoch)

    print('===> Finished !')

if __name__ == '__main__':
    main()