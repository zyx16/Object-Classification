from torch.utils.data import DataLoader

def create_dataset(opt):
    if opt['mode'] == 'oc': # Object classification
        from .dataset import PascalVOCDataset as d
        return d(opt)
    else:
        raise NotImplementedError('Dataset %s is not recognized.' % (mode))

def create_dataloader(ds, opt):
    if opt['phase'] == 'train':
        return DataLoader(ds, batch_size=opt['batch_size'], shuffle=True, num_workers=opt['num_workers'], pin_memory=True)
    else:
        if opt['batch_size']:
            return DataLoader(ds, batch_size=opt['batch_size'], shuffle=False, num_workers=opt['num_workers'], pin_memory=True)
        else:
            return DataLoader(ds, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)

