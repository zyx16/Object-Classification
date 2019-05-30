import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(1, 1, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
    
    
class SoftProposal(nn.Module):
    def __init__(self, N, max_iter, err_th, device=torch.device('cuda')):
        super(SoftProposal, self).__init__()
        self.N = N
        self.ep2 = (0.15 * N)**2
        self.device = device
        self.dist = self.init_dist().to(device)
        self.max_iter = max_iter
        self.err_th = err_th
        
    
    def init_dist(self):
        x = torch.arange(end=self.N)
        y = torch.arange(end=self.N)
        xx, yy = torch.meshgrid(x, y)
        grid = torch.stack((xx, yy), dim=0)
        dist = grid.reshape(2, -1, 1) - grid.reshape(2, 1, -1)
        dist = torch.exp(- torch.sum(dist ** 2, dim=0).float() / (2 * self.ep2))
        return dist
    
    def forward(self, feature):
        '''
        feature: B * K * N * N
        '''
        B, K, N, _ = feature.shape
        x = feature.reshape(-1, K, N * N, 1)
        y = feature.reshape(-1, K, 1, N * N)
        D_ = torch.norm(x - y, dim=1) * self.dist
        D = D_ / torch.sum(D_, dim=1).view(B, -1, 1)
        M = torch.ones(B, N * N, 1, dtype=torch.float, device=self.device) / (N * N)
        last_M = M
        for i in range(self.max_iter):
            M = torch.bmm(D, M)
            if torch.mean(torch.abs(M - last_M)) < self.err_th:
                break
            last_M = M
        return M.view(B, 1, N, N) * feature
