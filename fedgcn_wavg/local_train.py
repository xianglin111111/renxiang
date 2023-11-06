import math

import torch
import torch.nn.functional as F
from models import *
import copy
import os
import numpy as np
import matplotlib.pyplot as plt
from config import args_parser
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA

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

def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a

    return y

def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

def test(args, model, features, adj, labels, idx_test):
    model.eval()
    if args.FL == 'moon':
        _, output = model(features, adj)
    if args.FL == 'sample_fedavg':
        output = model(features, adj)
    else:
        output = model(features, adj)

    pred_labels = torch.argmax(output, axis=1)
    loss_test = F.nll_loss(output[idx_test], labels[idx_test])
    acc_test = accuracy(output[idx_test], labels[idx_test])

    # acc_test = metrics.accuracy_score(labels[idx_test].cpu().detach().numpy(), pred_labels[idx_test].cpu().detach().numpy())
    # f1_test=metrics.f1_score(labels[idx_test].cpu().detach().numpy(), pred_labels[idx_test].cpu().detach().numpy(),average='weighted')
    # auc_test=metrics.roc_auc_score(one_hot(labels[idx_test].cpu().detach().numpy()), output[idx_test].cpu().detach().numpy(),multi_class='ovr',average='weighted')

    return loss_test.item(), acc_test.item()  # , f1_test, auc_test


def Lhop_Block_matrix_train_fedavg(args, model, optimizer, features, adj, labels, communicate_index, in_com_train_data_index):
    model.train()
    optimizer.zero_grad()
    #print(features.shape)

    output = model(features[communicate_index], adj[communicate_index][:,communicate_index])
    loss_train = F.nll_loss(output[in_com_train_data_index], labels[communicate_index][in_com_train_data_index])
    acc_train = accuracy(output[in_com_train_data_index], labels[communicate_index][in_com_train_data_index])

    loss_train.backward()
    optimizer.step()
    optimizer.zero_grad()
    #print(loss_train,acc_train)
    return loss_train.item()



def Lhop_Block_matrix_train_fedprox(args, global_model, client_model, optimizer, features, adj, labels, communicate_index, in_com_train_data_index):
    client_model.train()
    optimizer.zero_grad()
    proximal_term = 0.0

    output = client_model(features[communicate_index], adj[communicate_index][:,communicate_index])

    for w, w_t in zip(client_model.parameters(), global_model.parameters()):
        proximal_term += (w - w_t).norm(2)

    loss_train = F.nll_loss(output[in_com_train_data_index], labels[communicate_index][in_com_train_data_index])
    acc_train = accuracy(output[in_com_train_data_index], labels[communicate_index][in_com_train_data_index])

    loss = loss_train + (args.mu / 2) * proximal_term
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()


def decay_log_fuc(x, base):
    y = 1 - np.log(x) / np.log(base) * 0.5
    return y

def decay_mi_fuc(x, mi):
    y = 1 - 0.5 * (mi ** x)
    return y

def L_hop_train(args, K, features, adj, labels, communicate_indexes, in_com_train_data_indexes, idx_val, down, clients_degree_sum, clients_degree):
    print('L_hop_train:   FL方法————————————{}'.format(args.FL))


    if args.dataset == 'cora' or args.dataset == 'citeseer' or args.dataset == 'pubmed':
        args.hidden = 16
        print('args.hidden', args.hidden)
    else:
        args.hidden = 256
        print('args.hidden', args.hidden)

    global_model = GCN(nfeat=features.shape[1],
                       nhid=args.hidden,
                       nclass=labels.max().item() + 1,
                       dropout=args.dropout,
                       NumLayers=args.num_layers)
    global_model.reset_parameters()
    global_model = global_model.cuda()

    clients_models = []
    for i in range(K):
        clients_models.append(GCN(nfeat=features.shape[1],
                                  nhid=args.hidden,
                                  nclass=labels.max().item() + 1,
                                  dropout=args.dropout,
                                  NumLayers=args.num_layers))
        clients_models[i] = clients_models[i].cuda()



    optimizers = []
    for i in range(K):
        optimizers.append(torch.optim.SGD(clients_models[i].parameters(),
                                    lr=args.lr, weight_decay=args.weight_decay))


    acc = []
    for t in range(args.iterations):
        acc_trains = []
        count_D = 0

        for i in range(K):
            clients_models[i].load_state_dict(global_model.state_dict())


        for i in range(K):
            count_k = in_com_train_data_indexes[i].shape[0]
            for epoch in range(args.epochs):
                # print('客户端{} 训练第{}轮'.format(i, epoch))
                if args.FL == 'fedprox':
                    # print('FL方法 ====================== Fedprox')
                    train_loss = Lhop_Block_matrix_train_fedprox(args, global_model, clients_models[i], optimizers[i],
                                                                features, adj, labels, communicate_indexes[i],
                                                                in_com_train_data_indexes[i])

                elif args.FL == 'fedavg':
                    # print('FL方法 ====================== FedAvg')
                    train_loss = Lhop_Block_matrix_train_fedavg(args, clients_models[i], optimizers[i],
                                                        features, adj, labels, communicate_indexes[i],
                                                        in_com_train_data_indexes[i])


                else:
                    exit('L_hop_train---未找到对应的FL训练方法')

        if args.avg == 'avg':
            FedAvg(global_model, clients_models, in_com_train_data_indexes)
        elif args.avg == 'degree_avg':
            degree_FedAvg(global_model, clients_models, in_com_train_data_indexes, clients_degree_sum)
        loss_val, acc_val = test(args, global_model, features, adj, labels, idx_val)

        acc.append(acc_val)
        print('FL--epoch:{}    acc_val:{}'.format(t, acc_val))


    max_acc = max(acc)
    print('最高准确率为：{}'.format(max_acc))
    if down:
        print('下载-----》')
        # path = './result_pos_fedprox'
        path = './result_noniid'
        if not os.path.exists(path):
            os.makedirs(path)
        if args.noniid_type == 'rate':
            rootpath = path + '/k{}_{}_{}_rate{}_{}_L-hop{}_maxacc{}.dat'.format(args.clients_num, args.dataset, args.avg, args.iid_percent, args.FL,
                                                                          args.L_hop, max_acc)
        elif args.noniid_type == 'LDA':
            rootpath = path + '/k{}_{}_{}_LDA{}_{}_L-hop{}_maxacc{}.dat'.format(args.clients_num, args.dataset, args.avg, args.non_alpha, args.FL,
                                                                         args.L_hop, max_acc)
        else:
            rootpath = None
            exit('ERROR--args.noniid_type 类型不正确')

        accfile = open(rootpath, 'w')
        for ac in acc:
            temp = str(ac)
            accfile.write(temp)
            accfile.write('\n')
        accfile.close()


def center_train(args, features, adj, labels, idx_val, down, idx_train):
    if args.dataset == 'cora' or args.dataset == 'citeseer' or args.dataset == 'pubmed':
        args.hidden = 16
        print('args.hidden', args.hidden)
    else:
        args.hidden = 256
        print('args.hidden', args.hidden)

    global_model = GCN(nfeat=features.shape[1],
                       nhid=args.hidden,
                       nclass=labels.max().item() + 1,
                       dropout=args.dropout,
                       NumLayers=args.num_layers)
    global_model.reset_parameters()
    global_model = global_model.cuda()

    optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    acc = []
    for t in range(args.iterations):

        in_train_sample_idx = []

        for epoch in range(args.epochs):
            # print('客户端{} 训练第{}轮'.format(i, epoch))

            global_model.train()
            optimizer.zero_grad()

            output = global_model(features, adj)

            loss_train = F.nll_loss(output[idx_train], labels[idx_train])
            loss_train.backward()
            optimizer.step()
            optimizer.zero_grad()
            # print(loss_train,acc_train)
            train_loss = loss_train.item()

        loss_val, acc_val = test(args, global_model, features, adj, labels, idx_val)

        acc.append(acc_val)
        print('FL--epoch:{}    acc_val:{}'.format(t, acc_val))

    max_acc = max(acc)
    print('最高准确率为：{}'.format(max_acc))
    if down:
        print('下载-----》')
        # path = './result_pos_fedprox'
        path = './result_center'
        if not os.path.exists(path):
            os.makedirs(path)

        rootpath = path + '/center_{}_localep{}_maxacc{}.dat'.format(args.dataset,args.epochs , max_acc)


        accfile = open(rootpath, 'w')
        for ac in acc:
            temp = str(ac)
            accfile.write(temp)
            accfile.write('\n')
        accfile.close()


# def local_train(args, features, adj, labels, idx_val, down, idx_train, in_train):
#     if args.dataset == 'cora' or args.dataset == 'citeseer' or args.dataset == 'pubmed':
#         args.hidden = 16
#         print('args.hidden', args.hidden)
#     else:
#         args.hidden = 256
#         print('args.hidden', args.hidden)
#
#     global_model = GCN(nfeat=features.shape[1],
#                        nhid=args.hidden,
#                        nclass=labels.max().item() + 1,
#                        dropout=args.dropout,
#                        NumLayers=args.num_layers)
#     global_model.reset_parameters()
#     global_model = global_model.cuda()
#
#     optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
#
#     acc = []
#     for t in range(args.iterations):
#
#         in_train_sample_idx = []
#
#         for epoch in range(args.epochs):
#             # print('客户端{} 训练第{}轮'.format(i, epoch))
#
#             train_loss = Lhop_Block_matrix_train_fedavg(args, global_model, optimizer,
#                                                         features, adj, labels, idx_train,
#                                                         in_train)
#
#
#
#         loss_val, acc_val = test(args, global_model, features, adj, labels, idx_val)
#
#         acc.append(acc_val)
#         print('FL--epoch:{}    acc_val:{}'.format(t, acc_val))
#
#     max_acc = max(acc)
#     print('最高准确率为：{}'.format(max_acc))
#     if down:
#         print('下载-----》')
#         # path = './result_pos_fedprox'
#         path = './result_local'
#         if not os.path.exists(path):
#             os.makedirs(path)
#
#         rootpath = path + '/center_{}_localep{}_maxacc{}.dat'.format(args.dataset,args.epochs , max_acc)
#
#
#         accfile = open(rootpath, 'w')
#         for ac in acc:
#             temp = str(ac)
#             accfile.write(temp)
#             accfile.write('\n')
#         accfile.close()


def local_train(args, K, features, adj, labels, communicate_indexes, in_com_train_data_indexes, idx_val, down, clients_degree_sum, clients_degree):
    print('L_hop_train:   FL方法————————————{}'.format(args.FL))


    if args.dataset == 'cora' or args.dataset == 'citeseer' or args.dataset == 'pubmed':
        args.hidden = 16
        print('args.hidden', args.hidden)
    else:
        args.hidden = 256
        print('args.hidden', args.hidden)

    global_model = GCN(nfeat=features.shape[1],
                       nhid=args.hidden,
                       nclass=labels.max().item() + 1,
                       dropout=args.dropout,
                       NumLayers=args.num_layers)
    global_model.reset_parameters()
    global_model = global_model.cuda()

    optimizer = torch.optim.SGD(global_model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


    acc = []
    for t in range(args.iterations):
        acc_trains = []
        count_D = 0



        for epoch in range(args.epochs):
            # print('客户端{} 训练第{}轮'.format(i, epoch))
            if args.FL == 'fedprox':
                print('FL方法 ====================== Fedprox')
                # train_loss = Lhop_Block_matrix_train_fedprox(args, global_model, clients_models[i], optimizers[i],
                #                                             features, adj, labels, communicate_indexes[i],
                #                                             in_com_train_data_indexes[i])

            elif args.FL == 'fedavg':
                # print('FL方法 ====================== FedAvg')
                train_loss = Lhop_Block_matrix_train_fedavg(args, global_model, optimizer,
                                                    features, adj, labels, communicate_indexes[4],
                                                    in_com_train_data_indexes[4])


            else:
                exit('L_hop_train---未找到对应的FL训练方法')

        loss_val, acc_val = test(args, global_model, features, adj, labels, idx_val)

        acc.append(acc_val)
        print('FL--epoch:{}    acc_val:{}'.format(t, acc_val))


    max_acc = max(acc)
    print('最高准确率为：{}'.format(max_acc))
    if down:
        print('下载-----》')
        # path = './result_pos_fedprox'
        path = './result_gcn'
        if not os.path.exists(path):
            os.makedirs(path)
        if args.noniid_type == 'rate':
            rootpath = path + '/k{}_{}_{}_rate{}_{}_L-hop{}_maxacc{}.dat'.format(args.clients_num, args.dataset, args.avg, args.iid_percent, args.FL,
                                                                          args.L_hop, max_acc)
        elif args.noniid_type == 'LDA':
            rootpath = path + '/k{}_{}_{}_LDA{}_{}_L-hop{}_maxacc{}.dat'.format(args.clients_num, args.dataset, args.avg, args.non_alpha, args.FL,
                                                                         args.L_hop, max_acc)
        else:
            rootpath = None
            exit('ERROR--args.noniid_type 类型不正确')

        accfile = open(rootpath, 'w')
        for ac in acc:
            temp = str(ac)
            accfile.write(temp)
            accfile.write('\n')
        accfile.close()


# # 节点加权聚合
# def FedAvg(global_model, clients_models, in_com_train_data_indexes):
#     states = []
#     gloabl_state = dict()
#     for i in range(len(clients_models)):
#         states.append(clients_models[i].state_dict())
#     # Average all parameters
#     for key in global_model.state_dict():
#         gloabl_state[key] = in_com_train_data_indexes[0].shape[0] * states[0][key]
#         count_D = in_com_train_data_indexes[0].shape[0]
#         for i in range(1, len(clients_models)):
#             gloabl_state[key] += in_com_train_data_indexes[i].shape[0] * states[i][key]
#             count_D += in_com_train_data_indexes[i].shape[0]
#         gloabl_state[key] /= count_D
#     global_model.load_state_dict(gloabl_state)


# 平均聚合
def FedAvg(global_model, clients_models, in_com_train_data_indexes):
    states = []
    gloabl_state = dict()
    for i in range(len(clients_models)):
        states.append(clients_models[i].state_dict())
    # Average all parameters
    for key in global_model.state_dict():
        gloabl_state[key] = states[0][key]
        count_D = 1
        for i in range(1, len(clients_models)):
            gloabl_state[key] += states[i][key]
            count_D += 1
        gloabl_state[key] /= count_D
    global_model.load_state_dict(gloabl_state)


# 节点的度加权聚合
def degree_FedAvg(global_model, clients_models, in_com_train_data_indexes, clients_degree_sum):

    print('每个客户端节点的度之和', clients_degree_sum)
    # print('每个客户端的节点数量', in_com_train_data_indexes)
    temp = []
    for i in range(len(in_com_train_data_indexes)):
        temp.append(in_com_train_data_indexes[i].shape[0])
    print('每个客户端节点数量', temp)

    states = []
    gloabl_state = dict()
    for i in range(len(clients_models)):
        states.append(clients_models[i].state_dict())

    # Average all parameters
    for key in global_model.state_dict():
        gloabl_state[key] = states[0][key] * clients_degree_sum[0]
        count_D = clients_degree_sum[0]
        for i in range(1, len(clients_models)):
            gloabl_state[key] += states[i][key] * clients_degree_sum[i]
            count_D += clients_degree_sum[i]
        gloabl_state[key] /= count_D
    global_model.load_state_dict(gloabl_state)

# # 节点的度加权聚合
# def degree_FedAvg(global_model, clients_models, in_com_train_data_indexes, clients_degree_sum):
#
#     # 先处理局部模型与全局模型之间的距离
#     term = []
#     for k in range(len(clients_models)):
#         proximal_term = 0
#         for w, w_t in zip(clients_models[k].parameters(), global_model.parameters()):
#             proximal_term += (w - w_t).norm(2)
#         term.append(proximal_term.item())
#
#     print('模型的L2范数为：', term)
#
#     # term = z_score(term)
#     #
#     # print('归一化后的L2范数为：', term)
#     #
#     # print('每个客户端节点的度之和', clients_degree_sum)
#     # temp = []
#     # for i in range(len(in_com_train_data_indexes)):
#     #     temp.append(in_com_train_data_indexes[i].shape[0])
#     # print('每个客户端节点数量', temp)
#
#     states = []
#     gloabl_state = dict()
#     for i in range(len(clients_models)):
#         states.append(clients_models[i].state_dict())
#
#     # Average all parameters
#     for key in global_model.state_dict():
#         gloabl_state[key] = states[0][key] * clients_degree_sum[0]  * term[0]
#         count_D = clients_degree_sum[0]  * term[0]
#         for i in range(1, len(clients_models)):
#             gloabl_state[key] += states[i][key] * clients_degree_sum[i]  * term[i]
#             count_D += clients_degree_sum[i]  * term[i]
#         gloabl_state[key] /= count_D
#     global_model.load_state_dict(gloabl_state)

# =========================================Server======================================================


