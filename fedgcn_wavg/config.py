import argparse


def args_parser():
    parser = argparse.ArgumentParser()
    # federated argument


    # FL优化方法参数
    parser.add_argument('--FL', type=str, default='fedavg', help="FL优化方法:fedprox fedavg")
    parser.add_argument('--dataset', type=str, default='reddit', help="数据集")    # 'cora', 'citeseer', 'pubmed', 'ogbn-arxiv', 'reddit'  , 'ogbn-products'
    parser.add_argument('--clients_num', type=int, default=82, help="客户端数量")     #  cora=7      citeseer= 6      pubmed=3      ogbn-arxiv=40       reddit= 41       ogbn-products=47
    parser.add_argument('--avg', type=str, default='degree_avg', help="avg  degree_avg")

    parser.add_argument('--iterations', type=int, default=300, help="FL迭代轮次")
    parser.add_argument('--epochs', type=int, default=3, help="本地训练迭代轮次")
    parser.add_argument('--down', type=bool, default=True, help="是否下载测试准确率  True  False")

    # Non-IID类型
    parser.add_argument('--noniid_type', type=str, default='rate', help="non-IID类型， rate  LDA")
    parser.add_argument('--iid_percent', type=float, default=1.0, help="rate中 独立同分布比率参数")       # rate = 1.0 的时候即为IID ，否则为Non-IID

    # GNN
    parser.add_argument('--num_layers', type=int, default=2, help="神经网络层数")
    parser.add_argument('--hidden', type=int, default=16, help="隐藏层神经元")
    parser.add_argument('--dropout', type=float, default=0.5, help="dropout")
    parser.add_argument('--lr', type=float, default=1.0, help="学习率")
    parser.add_argument('--weight_decay', type=float, default=5e-4, help="学习率衰减")

    # 图的优化方法参数
    parser.add_argument('--L_hop', type=int, default=1, help="FedGCN的L跳邻居采样，0跳即无修正")
    parser.add_argument('--sample', type=float, default=0.5, help="客户端训练节点采样比例  None")

    # fedprox  MOON  feddyn 参数
    parser.add_argument('--mu', type=float, default=0.01, help='FedProx 近端项参数')
    # parser.add_argument('--temperature', type=float, default=0.5, help='MOON温度参数')
    # parser.add_argument('--moon_mu', type=float, default=1.0, help='MOON 近端项参数')


    args = parser.parse_args()
    return args



