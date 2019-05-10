
def create_solver(opt):
    if opt['mode'] == 'oc':
        from .OCSolver import OCSolver as s
        return s(opt)
    else:
        raise NotImplementedError('Solver mode %s not implemented!' % (opt['mode']))
