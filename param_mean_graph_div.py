import torch
from torch_geometric.data import Data
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
from torch_geometric.datasets import KarateClub
# import torch_geometric.transforms as T
from torch_geometric.utils import to_networkx
import sys
import networkx as nx
from matplotlib import pyplot as plt
import numpy as np
import pymetis
import random

def main():
    # connection
    src = [0, 1, 2, 2, 2, 2, 3, 3, 4, 4, 5, 5, 5, 5, 6, 7, 8, 8,  8,  8, 9,  9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14]
    dst = [2, 2, 0, 1, 3, 8, 2, 4, 3, 5, 4, 6, 7, 8, 5, 5, 2, 5,  9, 14, 8, 10,  9, 11, 10, 12, 11, 13, 12, 14,  8, 13]

    # edge 
    edge_index = torch.tensor([src, dst], dtype=torch.long)

    # features
    x = [[0]] * 15
    x = torch.tensor(x, dtype=torch.float)

    #label
    y = [[1], [0], [1], [0], [1], [1], [0], [1], [1], [0], [0], [0], [1], [1], [1]]
    y = torch.tensor(y, dtype=torch.float)

    data = Data(x=x, y=y, edge_index=edge_index)
    check_graph(data)
    
    gpart = GraphParatitionMetis()
    data_list = gpart.graph_partition(data, 2)

    
    # from torch_geometric.datasets import Planetoid
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # dataset = Planetoid(root='/tmp/Cora', name='Cora')
    # data = dataset[0].to(device)
    # print(data['edge_index'])




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
class GraphParatitionMetis:
    def graph_partition(self, data, graph_num):
        # graph_num is the number of graphs which input graph is devided to
        src = data['edge_index'][0].tolist()
        dst = data['edge_index'][1].tolist()
        print(len(src))

        # edge_list = [(src[i], dst[i]) for i in range(len(src))]
        # G = nx.Graph()
        # G.add_edges_from(edge_list)
        # print(G.number_of_nodes(), G.number_of_edges())
        # print(f"G is "
        #       f"{nx.node_connectivity(G)}-connected and "
        #       f"{nx.edge_connectivity(G)}-edge connected.")
        # sys.exit()


        # make node list
        node_list = []
        for node in src:
            if node not in node_list: node_list.append(node)
        node_list.sort()

        print("独立ノードを除去した後のノード数：", len(node_list))

        # relabeling node_index
        change_dic = {}
        undo_dic = {}
        for node in node_list:
            change_dic[node] = node_list.index(node)
            undo_dic[node_list.index(node)] = node


        # relabeling edge_index
        new_src, new_dst = [], []
        for k in range(len(src)):
            new_src.append(change_dic[src[k]])
            new_dst.append(change_dic[dst[k]])
         

        # identify which node each node is connected to
        connection_dic = {}
        for i in range(len(new_src)):
            if new_src[i] in connection_dic:
                connection_dic[new_src[i]].append(new_dst[i])
            else:
                connection_dic[new_src[i]] = [new_dst[i]]

        # create input file for pymetis
        node_num = len(connection_dic.keys())
        adjacency_list = []
        for i in range(node_num):
            if i in connection_dic:
                adjacency_list.append(connection_dic[i])
        
            
        
        n_cuts, membership = pymetis.part_graph(graph_num, adjacency=adjacency_list)

        nodes_parts = []
        max_node_num = 0
        for i in range(graph_num):
            nodes_part = np.argwhere(np.array(membership) == i).ravel()
            # random.shuffle(nodes_part)
            # print("nodes_part: \n",nodes_part)
            nodes_parts.append(nodes_part)
            max_node_num = max(max_node_num, len(nodes_part))


        # subgraph expansion
        src_list = []
        dst_list = []
        node_lists = []
        for i in range(graph_num):
            src_item = []
            dst_item = []
            node_list = []
            for src_node in nodes_parts[i]:
                if src_node not in node_list: node_list.append(src_node)
                for dst_node in connection_dic[src_node]:
                    if dst_node in nodes_parts[i]:
                        src_item.append(src_node)
                        dst_item.append(dst_node)
                        
                    if len(node_list) >= max_node_num: break
                if len(node_list) >= max_node_num: break
                    
            for src_node in nodes_parts[i]:
                for dst_node in connection_dic[src_node]:        
                    if dst_node not in nodes_parts[i]:
                        src_item.append(src_node)
                        dst_item.append(dst_node)
                        src_item.append(dst_node)
                        dst_item.append(src_node)

                        node_list.append(dst_node)

                    if len(node_list) >= max_node_num: break
                if len(node_list) >= max_node_num: break

        
            src_list.append(src_item)
            dst_list.append(dst_item)
            node_lists.append(node_list)
            print(f"node num: {len(node_list)}")
            print(f"edge num: {len(src_item)}")
            # for idx in range(len(src_item)): print(src_item[idx], dst_item[idx])

            
        
        # create data of each graph
        data_list = []
        x_list = data['x'].tolist()
        y_list = data['y'].tolist()
        change_dic_list = [{}] * graph_num
        for i in range(graph_num):

            # features, label
            x, y = [], []
            for node in node_lists[i]:
                x.append(x_list[undo_dic[node]])
                y.append(y_list[undo_dic[node]])
            x = torch.tensor(x, dtype=torch.float, requires_grad=True)
            y = torch.tensor(y, dtype=torch.float, requires_grad=True)
            

            # edge
            # relabeling node_index
            for node in node_lists[i]:
                change_dic_list[i][node] = node_lists[i].index(node)

            # relabeling edge_index
            new_src_list, new_dst_list = [], []
            for k in range(len(src_list[i])):
                new_src_list.append(change_dic_list[i][src_list[i][k]])
                new_dst_list.append(change_dic_list[i][dst_list[i][k]])
            edge_index = torch.tensor([new_src_list, new_dst_list], dtype=torch.long)

            # train_mask, test_mask
            train_mask = []
            test_mask = []
            for node in node_lists[i]:
                train_mask.append(data.train_mask[undo_dic[node]])
                test_mask.append(data.test_mask[undo_dic[node]])
            train_mask = torch.tensor(train_mask, dtype=torch.bool)
            test_mask = torch.tensor(test_mask, dtype=torch.bool)
            data_item = Data(x=x, y=y, edge_index=edge_index, train_mask=train_mask, test_mask=test_mask)
            data_list.append(data_item)
        return change_dic_list, data_list, nodes_parts
            


        

 
if __name__ == "__main__":
    main()