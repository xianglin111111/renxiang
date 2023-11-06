import torch.nn as nn
import torch.nn.functional as F
import torch
from typing import Optional
from torch import Tensor
from torch.nn import Parameter

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn.inits import zeros
from torch_geometric.typing import (
    Adj,
    OptPairTensor,
    OptTensor,
    SparseTensor,
    torch_sparse,
)
from torch_geometric.utils import add_remaining_self_loops
from torch_geometric.utils import add_self_loops as add_self_loops_fn
from torch_geometric.utils import (
    is_torch_sparse_tensor,
    scatter,
    spmm,
    to_edge_index,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.utils.sparse import set_sparse_value

from torch_geometric.nn import GCNConv, GATConv

# from GCN_normal import GCNConv




# class GCN(torch.nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout, NumLayers):
#         super(GCN, self).__init__()
#
#         self.convs = torch.nn.ModuleList()
#         self.convs.append(
#             GCNConv(nfeat, nhid, normalize=True))
#         for _ in range(NumLayers - 2):
#             self.convs.append(
#                 GCNConv(nhid, nhid, normalize=True))
#         self.convs.append(
#             GCNConv(nhid, nclass, normalize=True))
#
#         self.dropout = dropout
#
#     def reset_parameters(self):
#         for conv in self.convs:
#             conv.reset_parameters()
#
#     def forward(self, x, adj_t):
#
#         for conv in self.convs[:-1]:
#             x = conv(x, adj_t)
#             x = F.relu(x)
#             x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.convs[-1](x, adj_t)
#         return torch.log_softmax(x, dim=-1)



class GCN(torch.nn.Module):
    def __init__(self, nfeat, nhid, nclass, dropout, NumLayers):
        super(GCN, self).__init__()

        self.convs = torch.nn.ModuleList()
        self.convs.append(
            GCNConv(nfeat, nhid, normalize=True, cached=True))
        for _ in range(NumLayers - 2):
            self.convs.append(
                GCNConv(nhid, nhid, normalize=True, cached=True))
        self.convs.append(
            GCNConv(nhid, nclass, normalize=True, cached=True))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, adj_t):

        for conv in self.convs[:-1]:
            x = conv(x, adj_t)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t)
        return torch.log_softmax(x, dim=-1)



# class GAT(torch.nn.Module):
#     def __init__(self, nfeat, nhid, nclass, dropout, NumLayers):
#         super(GAT, self).__init__()
#
#         self.convs = torch.nn.ModuleList()
#         self.convs.append(
#             GATConv(nfeat, nhid, heads=8, dropout=dropout))
#
#         self.convs.append(
#             GATConv(nhid * 8, nclass, dropout=dropout))
#
#         self.dropout = dropout
#
#
#     def reset_parameters(self):
#         for conv in self.convs:
#             conv.reset_parameters()
#
#     def forward(self, x, adj_t):
#         # for conv in self.convs[:-1]:
#         #     x = conv(x, adj_t)
#         #     x = F.relu(x)
#         #     x = F.dropout(x, p=self.dropout, training=self.training)
#         # x = self.convs[-1](x, adj_t)
#         # return torch.log_softmax(x, dim=-1)
#
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = F.elu(self.convs[0](x, adj_t))
#         x = F.dropout(x, p=self.dropout, training=self.training)
#         x = self.convs[-1](x, adj_t)
#         return torch.log_softmax(x, dim=-1)
