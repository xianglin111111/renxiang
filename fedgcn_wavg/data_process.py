#setting of data generation

import torch
import random
import sys
import pickle as pkl
import numpy as np
import scipy.sparse as sp
#from scipy.sparse.linalg.eigen.arpack import eigsh
import networkx as nx
import torch_geometric
import torch_sparse
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
from torch_geometric.datasets import Reddit


def generate_data(number_of_nodes, class_num, link_inclass_prob, link_outclass_prob):
    
    
    adj=torch.zeros(number_of_nodes,number_of_nodes) #n*n adj matrix

    labels=torch.randint(0,class_num,(number_of_nodes,)) #assign random label with equal probability
    labels=labels.to(dtype=torch.long)
    #label_node, speed up the generation of edges
    label_node_dict=dict()

    for j in range(class_num):
            label_node_dict[j]=[]

    for i in range(len(labels)):
        label_node_dict[int(labels[i])]+=[int(i)]


    #generate graph
    for node_id in range(number_of_nodes):
                j=labels[node_id]
                for l in label_node_dict:
                    if l==j:
                        for z in label_node_dict[l]:  #z>node_id,  symmetrix matrix, no repeat
                            if z>node_id and random.random()<link_inclass_prob:
                                adj[node_id,z]= 1
                                adj[z,node_id]= 1
                    else:
                        for z in label_node_dict[l]:
                            if z>node_id and random.random()<link_outclass_prob:
                                adj[node_id,z]= 1
                                adj[z,node_id]= 1
                              
    adj=torch_geometric.utils.dense_to_sparse(torch.tensor(adj))[0]

    #generate feature use eye matrix
    features=torch.eye(number_of_nodes,number_of_nodes)
    
    
    

    #seprate train,val,test
    idx_train = torch.LongTensor(range(number_of_nodes//5))
    idx_val = torch.LongTensor(range(number_of_nodes//5, number_of_nodes//2))
    idx_test = torch.LongTensor(range(number_of_nodes//2, number_of_nodes))



    return features.float(), adj, labels, idx_train, idx_val, idx_test
    


def parse_index_file(filename):
    """分析索引文件."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index
  
    

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx



def load_data(dataset_str):
    """
    Loads input data from gcn/data directory

    以cora为例子：dataset = cora
    ind.dataset_str.x =>  训练集节点特征向量，保存对象为：scipy.sparse.csr.csr_matrix，实际展开后大小为： (140, 1433);
    ind.dataset_str.tx => 测试集节点特征向量，保存对象为：scipy.sparse.csr.csr_matrix，实际展开后大小为： (1000, 1433);
    ind.dataset_str.allx => 包含有标签和无标签的训练节点特征向量，保存对象为：scipy.sparse.csr.csr_matrix，
                            实际展开后大小为：(1708, 1433)，可以理解为除测试集以外的其他节点特征集合，训练集是它的子集;
    ind.dataset_str.y => one-hot表示的训练节点的标签，保存对象为：numpy.ndarray;
    ind.dataset_str.ty => one-hot表示的测试节点的标签，保存对象为：numpy.ndarray;
    ind.dataset_str.ally => one-hot表示的ind.cora.allx对应的标签，保存对象为：numpy.ndarray;
    ind.dataset_str.graph => 保存节点之间边的信息，保存格式为：{ index : [ index_of_neighbor_nodes ] };
    ind.dataset_str.test.index => 保存测试集节点的索引，保存对象为：List，用于后面的归纳学习设置.

    All objects above must be saved using python pickle module.

    :param dataset_str: Dataset name
    :return: All data input files loaded (as well the training/test data).
    """
    if dataset_str in ['cora', 'citeseer', 'pubmed']:
        names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
        objects = []
        for i in range(len(names)):
            with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
                if sys.version_info > (3, 0):
                    objects.append(pkl.load(f, encoding='latin1'))      # 反序列化操作，读取文件
                else:
                    objects.append(pkl.load(f))      # 反序列化操作，读取文件


        x, y, tx, ty, allx, ally, graph = tuple(objects)
        test_idx_reorder = parse_index_file("data/ind.{}.test.index".format(dataset_str))
        test_idx_range = np.sort(test_idx_reorder)

        if dataset_str == 'citeseer':
            # Fix citeseer dataset (there are some isolated nodes in the graph)
            # Find isolated nodes, add them as zero-vecs into the right position
            test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
            tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
            tx_extended[test_idx_range-min(test_idx_range), :] = tx
            tx = tx_extended
            ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
            ty_extended[test_idx_range-min(test_idx_range), :] = ty
            ty = ty_extended

        features = sp.vstack((allx, tx)).tolil()    # 对稀疏矩阵的元素值做大量访问时，首先将待访问的稀疏矩阵进行 sp.tolil(), 能加快速度。
        features[test_idx_reorder, :] = features[test_idx_range, :]

        adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

        labels = np.vstack((ally, ty))
        labels[test_idx_reorder, :] = labels[test_idx_range, :]

        number_of_nodes=adj.shape[0]


        idx_test = test_idx_range.tolist()
        idx_train = range(len(y))
        idx_val = range(len(y), len(y)+500)

        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)

        print('idx_train', idx_train)

        #features = normalize(features) #cannot converge if use SGD, why??????????
        #adj = normalize(adj)    # no normalize adj here, normalize it in the training process


        features=torch.tensor(features.toarray()).float()
        adj = torch.tensor(adj.toarray())
        # print('adj---------', adj)
        adj = torch_sparse.tensor.SparseTensor.from_edge_index(torch_geometric.utils.dense_to_sparse(adj)[0])
        # print('adj***********', adj)
        #edge_index=torch_geometric.utils.dense_to_sparse(torch.tensor(adj.toarray()))[0]
        labels=torch.tensor(labels)
        labels=torch.argmax(labels,dim=1)
    elif dataset_str in ['ogbn-arxiv', 'ogbn-products', 'ogbn-mag', 'ogbn-papers100M']: #'ogbn-mag' is heteregeneous
        #from ogb.nodeproppred import NodePropPredDataset
        from ogb.nodeproppred import PygNodePropPredDataset

        # Download and process data at './dataset/.'

        if dataset_str == 'ogbn-arxiv':
            #dataset = NodePropPredDataset(name = dataset_str, root = 'dataset/')
            dataset = PygNodePropPredDataset(name='ogbn-arxiv',root= './data',
                                         transform=torch_geometric.transforms.ToSparseTensor())
        elif dataset_str == 'ogbn-products':
            #dataset = NodePropPredDataset(name = dataset_str, root = 'dataset/')
            dataset = PygNodePropPredDataset(name='ogbn-products',root= './data',
                                         transform=torch_geometric.transforms.ToSparseTensor())
        else:
            dataset = None
            exit('dataset_str is not in [ogbn-arxiv, ogbn-products, ogbn-mag, ogbn-papers100M]')


        split_idx = dataset.get_idx_split()
        idx_train, idx_val, idx_test = split_idx["train"], split_idx["valid"], split_idx["test"]
        print('idx_train', idx_train)
        idx_train = torch.LongTensor(idx_train)
        idx_val = torch.LongTensor(idx_val)
        idx_test = torch.LongTensor(idx_test)
        data = dataset[0]

        features = data.x #torch.tensor(graph[0]['node_feat'])
        labels = data.y.reshape(-1) #torch.tensor(graph[1].reshape(-1))
        adj = data.adj_t.to_symmetric()
        #edge_index = torch.tensor(graph[0]['edge_index'])
        #adj = torch_geometric.utils.to_dense_adj(torch.tensor(graph[0]['edge_index']))[0]
    elif dataset_str == 'reddit':
        dataset = Reddit(root='./dataset/reddit')
        data = dataset[0]

        idx_train = torch.nonzero(data.train_mask).squeeze()
        idx_val = torch.nonzero(data.val_mask).squeeze()
        idx_test = torch.nonzero(data.test_mask).squeeze()

        features = data.x  # torch.tensor(graph[0]['node_feat'])
        labels = data.y.reshape(-1)  # torch.tensor(graph[1].reshape(-1))

        edge_index = data.edge_index
        adj = SparseTensor(row=edge_index[0], col=edge_index[1])


    return features.float(), adj, labels, idx_train, idx_val, idx_test


def load_graph(dataset_str):

    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open("data/ind.{}.{}".format(dataset_str, names[i]), 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))  # 反序列化操作，读取文件
            else:
                objects.append(pkl.load(f))  # 反序列化操作，读取文件

    x, y, tx, ty, allx, ally, graph = tuple(objects)

    return y, graph

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)

def subgraph_sample(graph, idx_train, idx_val, idx_test):
    '''
    graph: the original graph
    idx_train: the index of training set nodes in the graph
    idx_val: the index of validation set nodes in the graph
    idx_test: the index of test set nodes in the graph
    '''
    size_of_graph = len(graph)


    target_size = 1500
    number_subgraph = 1  # -----------------------------------------------------------原本50

    num_train = idx_train.size()[0]
    num_val = idx_val.size()[0]
    num_test = idx_test.size()[0]

    print('num_train', num_train)

    subgraph_set_dict = {}
    print('Target number of subgraph:', number_subgraph)
    for iter in range(number_subgraph):
        #select initial node, and store it in the index_subgraph list
        index_subgraph = [np.random.randint(0, num_train)]      # 中心节点的索引

        #the neighbor node set of the initial nodes
        neighbors = graph[index_subgraph[0]]                    # 中心节点的邻居索引
        print('index_subgraph==============', neighbors)
        len_subgraph = 0
        while(1):
            len_neighbors = len(neighbors)
            if(len_neighbors == 0):# 卡在互连图中，选择重新启动节点
                while(1):
                    restart_node = np.random.randint(0, num_train)
                    if(restart_node not in index_subgraph):
                        break
                index_subgraph.append(restart_node)
                neighbors = neighbors + graph[restart_node]
                neighbors = list(set(neighbors) - set(index_subgraph))
            else:
                # 选择部分（一半）邻居节点并将其插入到当前子图中
                if ((target_size - len_subgraph) > (len_neighbors*0.5)):        # 判断我们是否需要选择那么多邻居
                    neig_random = random.sample(neighbors, max(1, int(0.5*len_neighbors)))
                    neighbors = list(set(neighbors) - set(neig_random))     # 选出set(neighbors)中与 neig_random 不重复的元素

                    index_subgraph = index_subgraph + neig_random
                    index_subgraph = list(set(index_subgraph))
                    for i in neig_random:
                        neighbors = neighbors + graph[i]
                    neighbors = list(set(neighbors) - set(index_subgraph))
                    len_subgraph = len(index_subgraph)
                else:
                    neig_random = random.sample(neighbors, (target_size - len_subgraph))
                    index_subgraph = index_subgraph + neig_random
                    index_subgraph = list(set(index_subgraph))

                    break

        idx_train_subgraph = []
        idx_val_subgraph = []
        idx_test_subgraph = []
        idx_test_dict = {}
        print('index_subgraph前', len(index_subgraph))
        print('list(range(num_train, num_train+num_val))', list(range(num_train, num_train+num_val)))
        print('index_subgraph', index_subgraph)
        index_subgraph = list(set(index_subgraph + list(range(num_train, num_train+num_val))))  # 将子图索引中，验证集未包含的节点加入。

        print('index_subgraph后', len(index_subgraph))
        #record the new index of nodes in the subgraph
        for i in range(len(index_subgraph)):
            if(index_subgraph[i]<num_train):
                idx_train_subgraph.append(i)
            elif(index_subgraph[i]<(num_train + num_val)):
                idx_val_subgraph.append(i)
            elif(index_subgraph[i]<(num_train + num_val + num_test)):
                idx_test_subgraph.append(i)
                idx_test_dict[i] = index_subgraph[i]

        print(iter + 1, 'th subgraph has been sampled')

        #generate the adjacency matrix of the subgraph
        G = nx.from_dict_of_lists(graph)
        g = G.subgraph(index_subgraph)
        adj =nx.adjacency_matrix(g)
        print('前 adj', adj)
        adj = normalize_adj(adj + sp.eye(adj.shape[0]))
        print('后 adj', adj)

        index_subgraph = torch.LongTensor(index_subgraph)
        idx_train_subgraph = torch.LongTensor(idx_train_subgraph)
        idx_val_subgraph = torch.LongTensor(idx_val_subgraph)
        idx_test_subgraph = torch.LongTensor(idx_test_subgraph)

        #store the information of generated subgraph:
        #indices of nodes in the original graph G;
        #adjacency matrix;
        #new indices (indices in the subgraph) of nodes belong to train, val, test set.
        #In this way, we do not have to load the adjacency matrix of the original graph during the training process
        subgraph_set_dict[iter] = {'index_subgraph': index_subgraph, 'adj':adj,
                 'idx_train': idx_train_subgraph, 'idx_val': idx_val_subgraph,
                 'idx_test': idx_test_subgraph, 'idx_test_dict':idx_test_dict}

    return subgraph_set_dict


if __name__ == '__main__':
    # features, adj, labels, idx_train, idx_val, idx_test = load_data('cora')
    # # features.shape torch.Size([2708, 1433])
    # # labels.shape torch.Size([2708])
    #
    # #==========================================================================================
    # # adj SparseTensor(row=tensor([   0,    0,    0,  ..., 2707, 2707, 2707]),
    # #              col=tensor([ 633, 1862, 2582,  ...,  598, 1473, 2706]),
    # #              size=(2708, 2708), nnz=10556, density=0.14%)
    # # row 非零值的行下标，即edge_index[0]    col:非零值的列下标，即edge_index[1]      size:稀疏矩阵大小
    # #==========================================================================================
    #
    # # idx_train.shape torch.Size([140])
    # # idx_val.shape torch.Size([500])
    # # idx_test.shape torch.Size([1000])
    #
    # row, col, edge_attr = adj.t().coo()
    # print(adj)
    # # print('-----------')
    # # print(row)
    # # a = int(2)
    # # a = torch.tensor(a)
    # # row[0] = a
    # # print('row', row)
    #
    # print(adj[0])
    # print('*'*100)



    # y, graph = load_graph('cora')
    # idx_train = range(len(y))
    # idx_val = range(len(y), len(y)+500)
    # idx_test = range(len(y) + 500, len(y) + 1500)
    # idx_train = torch.LongTensor(idx_train)
    # idx_val = torch.LongTensor(idx_val)
    # idx_test = torch.LongTensor(idx_test)
    #
    # sampled_subgraph_dict = subgraph_sample(graph,idx_train,idx_val,idx_test)
    # # print('sampled_subgraph_dict', sampled_subgraph_dict[0])
    # print('sampled_subgraph_dict', len(sampled_subgraph_dict[0]['index_subgraph']))

    features, adj, lable, train, val, test, graph = load_data('cora')

    # print('graph', graph)
    # print(train)
    print(adj)
    a = adj.storage.colcount()

    row, col, _ = adj.coo()
    deg_in = 1. / adj.storage.colcount()
    deg_out = 1. / adj.storage.rowcount()

    prob = (1. / deg_in[row]) + (1. / deg_out[col])

    print(prob)
    print(len(prob))












