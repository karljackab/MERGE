import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from collections import defaultdict
import random

class Merge_Dataset(Dataset):
    def __init__(self, dataset, train_data_ratio=1., params=(4, 9, 10, 10)):
        self.train_data_ratio = train_data_ratio
        self.l0, self.l1, self.tao, self.l3 = params   ## l0: context window size, l1: # neg samples, tao: sample path len, l3: # pths/node
        self.NODE_NUM = 0
        self.adj_list = defaultdict(list) ## key=node_id, value=list of adj_node_id
        self.label_idx_map = dict()
        self.node_id_mapping = dict()
        self.edge_num = 0

        if dataset == 'cora':
            self.NODE_NUM = 2708
            self.raw_cont = torch.zeros((self.NODE_NUM, 1433))

            self.adj_matrix = torch.zeros((self.NODE_NUM, self.NODE_NUM))

            with open('dataset/cora/cora.content') as f:
                for row in f.readlines():
                    cur_cont = row.strip().split('\t')
                    cur_node_id, label = int(cur_cont[0]), cur_cont[-1]
                    cur_cont = list(map(lambda x: int(x), cur_cont[1:-1]))

                    if cur_node_id not in self.node_id_mapping:
                        self.node_id_mapping[cur_node_id] = len(self.node_id_mapping)
                    if label not in self.label_idx_map:
                        self.label_idx_map[label] = len(self.label_idx_map)
                    self.raw_cont[self.node_id_mapping[cur_node_id]] = torch.tensor(cur_cont)
                
                with open('dataset/cora/cora.cites') as f:
                    for row in f.readlines():
                        a, b = map(lambda x: self.node_id_mapping[int(x)], row.strip().split('\t'))
                        self.adj_list[b].append(a)
                        self.adj_matrix[b][a] = 1.
                        self.edge_num += 1
            
        else:
            raise Exception("Not support dataset!")



        assert self.NODE_NUM > 0 
        self.adj_keys = list(self.adj_list.keys())
        self.deg_ratio = list(map(lambda x: len(x)/self.NODE_NUM, self.adj_list.values()))
        assert type(self.raw_cont) == torch.Tensor
        self.train_node_list = random.sample(self.adj_keys, k=int(train_data_ratio*len(self.adj_keys))) ## some nodes have no edges
        # print(self.train_node_list)
    
    def cal_two_nodes_distance(self, node_s, node_t):
        if node_s == node_t:
            return 0

        dist_list = [(node_s, 0)]
        res = None
        while len(dist_list) > 0:
            cur_node, cur_dist = dist_list[0]
            del dist_list[0]

            for next_node in self.adj_list[cur_node]:
                if next_node == node_t:
                    res = cur_dist+1
                    break
                if cur_dist < self.l0:
                    dist_list.append((next_node, cur_dist+1))
            if res is not None:
                break
        return res

    def random_walk(self, node_s):
        res = set()
        for _ in range(self.l3):
            cur_node = node_s
            for _ in range(self.tao):
                if len(self.adj_list[cur_node]) == 0:
                    continue
                cur_node = random.choice(self.adj_list[cur_node])
                distance = self.cal_two_nodes_distance(node_s, cur_node)
                if distance is not None:
                    res.add(cur_node)
        return list(res)
    def neg_sample(self):
        return random.choices(
            population=self.adj_keys,
            weights=self.deg_ratio,
            k=self.l1
        )

    def __getitem__(self, idx):
        pos_nodes_list = self.random_walk(self.train_node_list[idx])
        neg_nodes_list = self.neg_sample()

        labels = torch.tensor([1]*len(pos_nodes_list) + [-1]*len(neg_nodes_list))
        nodes_cont = torch.cat((self.raw_cont[pos_nodes_list], self.raw_cont[neg_nodes_list]))
        nodes_adj = torch.cat((self.adj_matrix[pos_nodes_list], self.adj_matrix[neg_nodes_list]))

        return nodes_adj, nodes_cont, labels

    def __len__(self):
        return len(self.train_node_list)
