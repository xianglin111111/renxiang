import torch
from data_process import load_data
import numpy as np
import torch_geometric
import random

from models import *
from config import args_parser
from local_train import *
from collections import Counter






def setdiff1d(t1, t2):
    combined = torch.cat((t1, t2))
    # t1.shape: torch.Size([2708])
    # t2.shape: torch.Size([1353])
    # combined.shape: torch.Size([4061])

    uniques, counts = combined.unique(return_counts=True)   # uniques:返回不重复的元素，并按照顺序  counts:每个不重复元素的数量
    difference = uniques[counts == 1]   # 返回的是没有重复的元素，即不同标签下标
    # intersection = uniques[counts > 1]
    return difference

def intersect1d(t1, t2):
    combined = torch.cat((t1, t2))
    uniques, counts = combined.unique(return_counts=True)
    # difference = uniques[counts == 1]
    intersection = uniques[counts > 1]      # 返回的是重复的元素的下标
    return intersection



def dataset_cut(K, adj, labels, iid_percent):
    # row, col, edge_attr = adj.t().coo()     # .coo()  从adj中取行，列，边权重
    # edge_index = torch.stack([row, col], dim=0)

    # print('edge_index', edge_index)

    nclass = labels.max().item() + 1        # 类的数量
    split_data_indexes = []                 # 切分数据的索引



    extar_num = K % nclass
    split_num = K // nclass


    non_iid_percent = 1 - float(iid_percent)
    print('Non-IID 比例', non_iid_percent)
    iid_indexes = []  # random assign
    class_labels = []  # 将训练数据点拆分为不同的设备


    # 类别数量=客户端数量
    class_data_n = []
    for i in range(nclass):
        current = torch.nonzero(labels == i).reshape(-1)        # 非零元素定位, True即非零，即返回label=i的元素坐标。
        # print('k {}, current, {}'.format(i, current))
        current = current[np.random.permutation(len(current))]  # shuffle
        class_labels.append(current)
        class_data_n.append(len(class_labels[i]))
        print('标签：{}     数据数量{}'.format(i, len(class_labels[i])))


    if extar_num == 0:
        max_num_client = None
    else:
        max_num_client = np.argsort(class_data_n)[-extar_num:]


    shuffle_labels = []
    for i in range(nclass):
        if max_num_client is None:
            num = split_num

        else:
            if i in max_num_client:
                num = split_num + 1
            else:
                num = split_num

        temp = np.array_split(class_labels[i], num)
        shuffle_labels += temp



    # 为non-IID划分数组，non-IID数组中，只包含一种标签
    for i in range(K):
        label_i = i
        labels_class = shuffle_labels[label_i]

        average_num = int(len(labels_class) * non_iid_percent)
        split_data_indexes.append((labels_class[0:average_num]))     # 按照比例，取相同类别前百分比作为非独立同分布划分数组。

        # print('标签：{}     切分好的相同类别的长度：{}'.format(i, len(split_data_indexes[i])))

    iid_indexes = setdiff1d(torch.tensor(range(len(labels))).cuda(), torch.cat(split_data_indexes))     # IID节点的索引列表
    iid_indexes = iid_indexes[np.random.permutation(len(iid_indexes))]
    print('iid_indexes', len(iid_indexes))
    print('-' * 50)

    for i in range(K):  # for iid
        label_i = i
        labels_class = shuffle_labels[label_i]
        average_num = int(len(labels_class) * (1 - non_iid_percent))
        if (i+1) == K:
            split_data_indexes[i] = list(split_data_indexes[i]) + list(iid_indexes)
        else:
            split_data_indexes[i] = list(split_data_indexes[i]) + list(iid_indexes[:average_num])
        iid_indexes = iid_indexes[average_num:]

        print('客户端：{}     客户端数据的长度：{}'.format(i, len(split_data_indexes[i])))

    return split_data_indexes

def dirichlet_cut(args, K, labels, idx_train, iid_percent):
    print('label', labels)

    nclass = labels.max().item() + 1        # 类的数量
    # (K, N)的类别标签分布矩阵X，记录每个client占有每个类别的多少
    label_distribution = np.random.dirichlet([args.non_alpha] * K, nclass)

    # 记录每个K个类别对应的样本下标  class_idcs[0]->都是0标签的下标    class_idcs[1]->都是1标签的下标
    class_idcs = [torch.argwhere(labels == y).flatten()
                  for y in range(nclass)]
    in_idcs = [torch.argwhere(labels[idx_train] == y).flatten()
                  for y in range(nclass)]

    train_node = []
    last_node = []
    for i in range(len(class_idcs)):
        chongfu = setdiff1d(class_idcs[i], in_idcs[i])
        train_node.append(chongfu)
        last_node.append(in_idcs[i])


    # print('标签下标：class_idcs', class_idcs[1])

    client_idcs = [[] for _ in range(K)]        # 为客户端生成新的空数组
    client_idcs_train = [[] for _ in range(K)]  # 为客户端生成新的空数组
    # 记录N个client分别对应样本集合的索引
    for c, fracs in zip(last_node, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idcs 为遍历第i个client对应样本集合的索引
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs[i] += [idcs]


    for c, fracs in zip(train_node, label_distribution):
        # np.split按照比例将类别为k的样本划分为了N个子集
        # for i, idcs 为遍历第i个client对应样本集合的索引
        for i, idcs in enumerate(np.split(c, (np.cumsum(fracs)[:-1] * len(c)).astype(int))):
            client_idcs_train[i] += [idcs]

    for i in range(len(client_idcs)):
        for j in range(len(client_idcs[i])):
            temp_a = list(client_idcs[i][j].cpu().numpy())
            temp_b = list(client_idcs_train[i][j].cpu().numpy())
            temp = temp_a + temp_b
            temp = torch.tensor(temp).cuda()
            client_idcs[i][j] = temp

    client_idcs = [torch.concatenate(idcs) for idcs in client_idcs]

    clients = []    # client_idcs中存在浮点张量，转换整形给clients
    for i in range(len(client_idcs)):
        clients.append([])
        for k in range(len(client_idcs[i])):
            temp = int(client_idcs[i][k].cpu().numpy())
            clients[i].append(temp)
        clients[i] = torch.tensor(clients[i]).cuda()

    # for i in range(len(client_idcs)):
    #     for j in range(len(client_idcs[i])):
    #         print('j', client_idcs[i][j])
    #         temp = client_idcs[i][j].int()
    #         client_idcs[i][j] = temp
    #         print('j', client_idcs[i][j])

    return clients


# 返回：每个客户端包含所有节点的L跳的子图的节点索引，每个客户端用于训练的节点索引
def L_hop(K, clients, adj, idx_train, hop):
    communicate_indexes = []    # L跳节点的子图的节点索引
    in_com_train_data_indexes = []

    row, col, edge_attr = adj.t().coo()     # .coo()  从adj中取行，列，边权重
    edge_index = torch.stack([row, col], dim=0)

    for i in range(K):
        # print('clients[i]', clients[i])
        clients[i] = torch.tensor(clients[i])
        # print('clients[i]', clients[i])

        clients[i] = clients[i].sort()[0]       # 按照索引从小到大排序

        # clients[i]：求子图的节点     hop:跳数    edge_index:边索引
        communicate_index = torch_geometric.utils.k_hop_subgraph(clients[i], hop, edge_index)[0]   # 使用torch_geometric.utils.k_hop_subgraph提取客户端所有节点L跳节点的子图
        communicate_indexes.append(communicate_index)
        communicate_indexes[i] = communicate_indexes[i].sort()[0]

        inter = intersect1d(clients[i].cuda(), idx_train)  # 客户端中的训练节点。即客户端所有节点索引和训练节点索引重复的集合

        # 寻找训练数据在L跳节点的子图的节点索引的位置？
        in_com_train_data_indexes.append(
            torch.searchsorted(communicate_indexes[i], inter).clone())  # local id in block matrix
        # torch.searchsorted(x, y)：x 经过排序后，y 在 x 中寻找自己的每个元素的位置，输出维度与 y 保持一致。
        # .clone()  复制一个不共享内存且支持梯度传递与叠加的参数。

        if len(in_com_train_data_indexes[i]) == 0:
            exit('ERROR--func(L_hop)--客户端可用训练节点个数为0 ')
        # print('in_com_train_data_indexes', in_com_train_data_indexes[i])



    print('-'*50)

    for i in range(len(communicate_indexes)):
        print('client', i, '的训练节点的子图连接', communicate_indexes[i][in_com_train_data_indexes[i]])
        train_num = len(communicate_indexes[i][in_com_train_data_indexes[i]])
        print('训练节点数量', train_num)
    print('-'*50)

    # print('adj', adj)
    # print('包含L跳的客户端0的子图节点数量', len(communicate_indexes[0]))
    # print('客户端0用于训练的节点数量', len(in_com_train_data_indexes[0]))


    # print('communicate_indexes', communicate_indexes[0])
    # print('in_com_train_data_indexes', in_com_train_data_indexes[0])

    return communicate_indexes, in_com_train_data_indexes


def z_score(arr):
    normal = []
    for x in arr:
        x = float(x - np.mean(arr))/np.std(arr)
        normal.append(x)
    return normal

def min_max(arr):
    normal = []
    for x in arr:
        x = float(x - np.min(arr)) / (np.max(arr) - np.min(arr))
        normal.append(x)
    return normal

def get_degree(adj, communicate_indexes, in_com_train_data_indexes):
    clients_degree = []
    clients_degree_sum = []

    for i in range(len(communicate_indexes)):

        sub = adj[communicate_indexes[i]][:, communicate_indexes[i]]
        # 求in_com_train_data_indexes这几个节点在sub图中的度，度的和即为客户端参与训练边的和数
        degree = list(sub.storage.colcount()[in_com_train_data_indexes[i]].cpu().numpy())   # sub.storage.colcount() 求图中每个节点在列的度

        for k in range(len(degree)):
            degree[k] = float(degree[k])

        clients_degree.append(degree)
        clients_degree_sum.append(sum(degree))
    return clients_degree, clients_degree_sum


def softmax(x):
    e_x = np.exp(x - np.max(x))  # 防止指数溢出
    return e_x / np.sum(e_x)




if __name__ == '__main__':
    args = args_parser()


    torch.manual_seed(199)
    torch.cuda.manual_seed_all(199)
    random.seed(1)

    if args.noniid_type == 'LDA':
        print('LDA -> seed 1')
        np.random.seed(1)  #
    elif args.noniid_type == 'rate':
        print('rate -> seed 10')
        np.random.seed(100)  #
    else:
        np.random.seed(1)  #
    features, adj, labels, idx_train, idx_val, idx_test = load_data(args.dataset)

    class_num = labels.max().item() + 1
    clients_num = args.clients_num


    features = features.cuda()
    adj = adj.cuda()
    labels = labels.cuda()
    idx_train = idx_train.cuda()
    idx_val = idx_val.cuda()
    idx_test = idx_test.cuda()


    row, col, edge_attr = adj.t().coo()     # .coo()  从adj中取行，列，边权重
    edge_index = torch.stack([row, col], dim=0)


    if args.noniid_type == 'rate':
        clients = dataset_cut(K=clients_num, adj=adj, labels=labels, iid_percent=args.iid_percent)
    elif args.noniid_type == 'LDA':
        clients = dirichlet_cut(args=args, K=clients_num, labels=labels, idx_train=idx_train, iid_percent=args.iid_percent)
    else:
        clients = None
        exit('ERROR--args.noniid_type 类型不正确')

    # 计算每个客户端拥有的标签
    clients_label_num = []
    for i in range(clients_num):  # 每个客户端
        clients_label_num.append([])
        for j in range(class_num):  # 每个标签
            count = 0
            for z in range(len(clients[i])):    # 客户端中每个数据
                if labels[clients[i][z]] == j:
                    count += 1
            clients_label_num[i].append(count)
    print('每个客户端拥有的标签数量', clients_label_num)



    # print(adj.to_dense()) # 稀疏矩阵转换为正常矩阵
    # ----------------------------------------------------------------------------------------------------------------------------------------------------------------------
    communicate_indexes, in_com_train_data_indexes = L_hop(K=clients_num, clients=clients, adj=adj, idx_train=idx_train, hop=args.L_hop)

    for i in range(len(communicate_indexes)):
        print('客户端数据数量', len(communicate_indexes[i]))


    clients_degree, clients_degree_sum = get_degree(adj, communicate_indexes, in_com_train_data_indexes)     # 计算每个客户端每个训练节点的度 和 每个客户端所有训练节点的度的和

    for i in range(len(clients_degree)):
        print('客户端{}的度长度为：{}   度{}'.format(i, len(clients_degree[i]), clients_degree[i]))

    L_hop_train(args, K=clients_num, features=features, adj=adj, labels=labels, communicate_indexes=communicate_indexes,
                in_com_train_data_indexes=in_com_train_data_indexes, idx_val=idx_val, down=args.down, clients_degree_sum=clients_degree_sum, clients_degree=clients_degree)


    # local_train(args, K=clients_num, features=features, adj=adj, labels=labels, communicate_indexes=communicate_indexes,
    #             in_com_train_data_indexes=in_com_train_data_indexes, idx_val=idx_val, down=args.down, clients_degree_sum=clients_degree_sum, clients_degree=clients_degree)

    # center_train(args, features=features, adj=adj, labels=labels, idx_val=idx_val, down=args.down, idx_train=idx_train)