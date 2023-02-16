## GNNの実装 # パラメータの平均をとる
import torch
import torch.nn as nn

from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import KarateClub
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from param_mean_graph_div import GraphParatitionMetis
# from sugar_graph_div import GraphParatitionMetis
# from my_graph_div import GraphParatitionMetis
import numpy as np
import sys
import time
import statistics
import pandas as pd
import matplotlib.pyplot as plt

def check_graph(data):
    '''グラフ情報を表示'''
    print("グラフ構造:", data)
    print("グラフのキー: ", data.keys)
    print("ノード数:", data.num_nodes)
    print("エッジ数:", data.num_edges)
    print("ノードの特徴量数:", data.num_node_features)
    print("孤立したノードの有無:", data.contains_isolated_nodes())
    print("自己ループの有無:", data.contains_self_loops())
    print("====== ノードの特徴量:x ======")
    print(data['x'])
    print("====== ノードのクラス:y ======")
    print(data['y'])
    print("========= エッジ形状 =========")
    print(data['edge_index'])


class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        hidden_size = 15
        self.conv1 = GCNConv(dataset.num_node_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, dataset.num_classes)
    
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        
        return F.log_softmax(x, dim=1)



def test(data_list):
    # グラフの個数
    graph_num = len(data_list)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # model.train() #モデルを訓練モードにする。
    total_time = 0
    model_list = [GCN() for _ in range(graph_num)]
    optimizer_list = []
    for i in range(graph_num):
        data_list[i] = data_list[i].to(device)
        model_list[i] = model_list[i].to(device)
        model_list[i].train() #モデルを訓練モードにする。
        optimizer_list.append(torch.optim.Adam(model_list[i].parameters(), lr=0.01))



    loss_lists = [[] for _ in range(graph_num)]
    for epoch in range(400):
        if epoch < 400:
            for i in range(1, graph_num):
                model_list[0].conv1.lin.weight.data.add_(model_list[i].conv1.lin.weight.data)
            model_list[0].conv1.lin.weight.data.mul_(float(1.0) / graph_num)
            for i in range(1, graph_num):
                model_list[i].conv1.lin.weight.data = torch.clone(model_list[0].conv1.lin.weight.data)


            for i in range(1, graph_num):
                model_list[0].conv2.lin.weight.data.add_(model_list[i].conv2.lin.weight.data)
            model_list[0].conv2.lin.weight.data.mul_(float(1) / graph_num)
            for i in range(1, graph_num):
                model_list[i].conv2.lin.weight.data = torch.clone(model_list[0].conv2.lin.weight.data)

            
            for i in range(1, graph_num):
                model_list[0].conv1.bias.data.add_(model_list[i].conv1.bias.data)
            model_list[0].conv1.bias.data.mul_(float(1) / graph_num)
            for i in range(1, graph_num):
                model_list[i].conv1.bias.data = torch.clone(model_list[0].conv1.bias.data)

            for i in range(1, graph_num):
                model_list[0].conv2.bias.data.add_(model_list[i].conv2.bias.data)
            model_list[0].conv2.bias.data.mul_(float(1) / graph_num)
            for i in range(1, graph_num):
                model_list[i].conv2.bias.data = torch.clone(model_list[0].conv2.bias.data)
    
        for i in range(graph_num):
            for _ in range(100):
                time_sta = time.time()
                optimizer_list[i].zero_grad()
                out = model_list[i](data_list[i])
                yt = data_list[i].y.type(torch.LongTensor).to(device)
                loss = F.nll_loss(out[data_list[i].train_mask], yt[data_list[i].train_mask])
                loss_lists[i].append(loss.tolist())
                loss.backward()
                optimizer_list[i].step()
                time_end = time.time()
                total_time += time_end - time_sta
    # loss値の推移を保存
    # loss値の推移を保存
    # for i in range(graph_num):
    #     with open(f'experiment_data/loss_data/param_mean1/cora/fd_loss_graph_{graph_num}_{i+1}.txt', 'w') as f:
    #         for loss in loss_lists[i]:
    #             f.write(f"{loss}\n")

    acc_list = []
    for i in range(graph_num):
        model_list[i].eval() #モデルを評価モードにする。
        _, pred = model_list[i](data_list[i]).max(dim=1)
        correct = float(pred[data_list[i].test_mask].eq(data_list[i].y[data_list[i].test_mask]).sum().item())
        acc = correct / data_list[i].test_mask.sum().item()
        acc_list.append(acc)


    # correct_num = 0
    # for i in range(len(data_item['y'])):
    #     if pred[i] == data_item['y'][i]: correct_num += 1
    # acc = float(correct_num) / len(data_item['y'])
    print(total_time*1.25)
    return acc_list, total_time

from torch_geometric.datasets import Planetoid
from torch_geometric.datasets import NELL

torch.autograd.set_detect_anomaly(True)

dataset = Planetoid(root='/tmp/Cora', name="Cora")
data = dataset[0]
gpart = GraphParatitionMetis()
pair_distant_list = []
mean_acc_list = []
time_list = []
total_time = 0
for i in range(10):
    
    print(f"------------------{i+1}回目------------------")
    _, data_list, nodes_parts = gpart.graph_partition(data, 2)
    # pair_table = [[0] * 7 for _ in range(7)]
    # y1 = data_list[0].y.tolist()
    # y2 = data_list[1].y.tolist()
    # if len(y1) != len(y2): continue
    # for j in range(len(y1)):
    #     pair_table[int(y1[j])][int(y2[j])] += 1
    #     if int(y1[j])!=int(y2[j]):
    #         pair_distant += 1
    # print(pair_distant)
        
    accuracies, time_part = test(data_list)
    total_time += time_part
    l = []
    for acc in accuracies:
        l.append(acc)
    print(accuracies)
    mean_acc = statistics.mean(l)
    # pair_distant_list.append(pair_distant)
    mean_acc_list.append(mean_acc)
    print(mean_acc)
    print()         

print(sum(mean_acc_list)/10)
print(total_time/8)
#     exp_data = pd.DataFrame({'nodes_part1': nodes_parts[0], 
#                             'y1': data_list[0].y.tolist(),
#                             'nodes_part2': nodes_parts[1], 
#                             'y2': data_list[1].y.tolist()})
#     exp_data.to_excel(f"experiment_data/acc_data/acc_{mean_acc}.xlsx")
#     pair_data = pd.DataFrame(pair_table)
#     pair_data.to_csv(f"experiment_data/pair_data/acc_{mean_acc}.csv")
# df = pd.DataFrame({"精度": mean_acc_list}, index=pair_distant_list)
# df.to_csv('test.csv')
# print(df)
# plt.figure()
# df.plot()
# plt.savefig('experiment_data/pair_data/pair_distant.png')
# plt.close('all')
    # data_dic = {}
    # for i in [2]:
    #     for k in range(1, i+1):
    #         # dic_name = "experiment_data/loss_data/param_mean1/citeseer/"
    #         dic_name = "experiment_data/loss_data/param_mean1/cora/"
    #         filename = f"fd_loss_graph_{i}_{k}"
    #         path = dic_name + filename + ".txt"
    #         with open(path, 'r') as f:
    #             data = f.read().splitlines()
    #             data = [float(x) for x in data]
    #         data_dic[filename] = data

    # table = pd.DataFrame(index=range(1, 401), data=data_dic)
    # plt.figure()
    # table.plot()
    # plt.savefig('experiment_data/loss_data/param_mean1/loss_fig_citeseer.png')
    # plt.close('all')