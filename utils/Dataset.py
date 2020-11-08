import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data.dataset import Dataset
from collections import defaultdict
import random
import pickle
from tqdm import tqdm

def Merge_collfn(x):
    nodes_adj = torch.cat([x[i][0] for i in range(len(x))])
    nodes_cont = torch.cat([x[i][1] for i in range(len(x))])
    labels = torch.cat([x[i][2] for i in range(len(x))])

    source_nodes_cont = torch.stack([x[i][3] for i in range(len(x))])
    source_nodes_adj = torch.stack([x[i][4] for i in range(len(x))])

    cnt_list = list(map(lambda x: len(x[0]), x))
    source_mapping_idx = []
    for idx, cnt in enumerate(cnt_list):
        source_mapping_idx += [idx]*cnt
    return nodes_adj, nodes_cont, labels, source_nodes_cont, source_nodes_adj, source_mapping_idx

class Merge_Whole_Dataset(Dataset):
    def __init__(self, adj_matrix, raw_cont, label_list):
        self.adj_matrix, self.raw_cont, self.label_list = adj_matrix, raw_cont, label_list
        self.label_key_list = list(self.label_list.keys())
    def __getitem__(self, idx):
        new_idx = self.label_key_list[idx]
        return self.adj_matrix[new_idx], self.raw_cont[new_idx], self.label_list[new_idx]
    def __len__(self):
        return len(self.label_list)


class Merge_Dataset(Dataset):
    def __init__(self, dataset, train_data_ratio=1., params=(4, 9, 10, 10), suffix=None):
        self.train_data_ratio = train_data_ratio
        self.l0, self.l1, self.tao, self.l3 = params   ## l0: context window size, l1: # neg samples, tao: sample path len, l3: # pths/node
        self.NODE_NUM = 0
        self.adj_list = defaultdict(list) ## key=node_id, value=list of adj_node_id
        self.label_list = dict()
        self.label_idx_map = dict()
        self.node_id_mapping = dict()

        if dataset == 'cora' or dataset == 'citeseer':
            if dataset == 'cora':
                self.NODE_NUM = 2708
                self.raw_cont = torch.zeros((self.NODE_NUM, 1433))
                pca_node_dim, pca_net_dim = 650, 1150
            elif dataset == 'citeseer':
                self.NODE_NUM = 3312
                self.raw_cont = torch.zeros((self.NODE_NUM, 3703))
                pca_node_dim, pca_net_dim = 1300, 3000

            self.adj_matrix = torch.eye(self.NODE_NUM)

            for i in range(self.NODE_NUM):
                self.adj_list[i].append(i)

            with open(f'dataset/{dataset}/{dataset}.content') as f:
                for row in f.readlines():
                    cur_cont = row.strip().split('\t')
                    cur_node_id, label = cur_cont[0], cur_cont[-1]
                    cur_cont = list(map(lambda x: int(x), cur_cont[1:-1]))

                    if cur_node_id not in self.node_id_mapping:
                        self.node_id_mapping[cur_node_id] = len(self.node_id_mapping)
                    if label not in self.label_idx_map:
                        self.label_idx_map[label] = len(self.label_idx_map)
                    self.raw_cont[self.node_id_mapping[cur_node_id]] = torch.tensor(cur_cont)
                    # self.adj_list[self.node_id_mapping[cur_node_id]].append(self.node_id_mapping[cur_node_id])
                    self.label_list[self.node_id_mapping[cur_node_id]] = self.label_idx_map[label]

            if suffix == 'pca':
                print('Doing PCA at node content feature')
                pca_model = PCA(n_components=pca_node_dim, svd_solver='arpack')
                tmp_pca_raw_cont = pca_model.fit_transform(self.raw_cont.numpy())
                self.raw_cont = torch.from_numpy(tmp_pca_raw_cont.copy())
                del tmp_pca_raw_cont

            with open(f'dataset/{dataset}/{dataset}.cites') as f:
                for row in f.readlines():
                    try:
                        a, b = map(lambda x: self.node_id_mapping[x], row.strip().split('\t'))
                    except:
                        continue
                    self.adj_list[b].append(a)
                    self.adj_list[a].append(b)
                    self.adj_matrix[b][a] = 1.
                    self.adj_matrix[a][b] = 1.

            if suffix == 'pca':
                print('Doing PCA at adj feature')
                pca_model = PCA(n_components=pca_net_dim, svd_solver='arpack')
                tmp_pca_adj_matrix = pca_model.fit_transform(self.adj_matrix.numpy())
                self.adj_matrix = torch.from_numpy(tmp_pca_adj_matrix.copy())
                del tmp_pca_adj_matrix
        elif dataset == 'microblogPCU':
            self.NODE_NUM = 691
            self.raw_cont = torch.zeros((self.NODE_NUM, 3836))
            self.adj_matrix = torch.eye(self.NODE_NUM)
            for i in range(self.NODE_NUM):
                self.adj_list[i].append(i)
            with open('dataset/microblogPCU/new_user.pkl', 'rb') as f:
                users = pickle.load(f)
                for user in users:
                    self.raw_cont[user['user_id']] = torch.tensor(user['feature'])
                    if user['label'] != '':
                        self.label_list[user['user_id']] = 0 if int(user['label'])==-1 else 1
                    # self.adj_list[user['user_id']].append(user['user_id'])

            with open('dataset/microblogPCU/new_adj.csv', 'r') as f:
                for row in f.readlines():
                    a, b = row.strip().split(',')
                    a, b = int(a), int(b)
                    self.adj_list[b].append(a)
                    self.adj_list[a].append(b)
                    self.adj_matrix[b][a] = 1.
                    self.adj_matrix[a][b] = 1.
        else:
            raise Exception("Not support dataset!")

        assert self.NODE_NUM > 0 
        assert type(self.raw_cont) == torch.Tensor
        self.adj_keys = list(self.adj_list.keys())
        self.deg_ratio = list(map(lambda x: len(x)**0.75, self.adj_list.values()))

        self.train_node_list = random.sample(self.adj_keys, k=int(train_data_ratio*len(self.adj_keys)))

        ########################################## calculate distance matrix
        self.dist_mat = []
        for i in range(self.NODE_NUM):
            self.dist_mat.append([])
            for _ in range(self.NODE_NUM):
                self.dist_mat[-1].append(None)

        for i in tqdm(range(self.NODE_NUM)):
            self.cal_node_distance(i)
        for i in range(self.NODE_NUM):
            for j in range(self.NODE_NUM):
                if self.dist_mat[i][j] is not None and self.dist_mat[i][j] < 0:
                    self.dist_mat[i][j] = None

    def cal_node_distance(self, node_s):
        self.dist_mat[node_s][node_s] = 0
        dist_list = [(node_s, 0)]

        while len(dist_list) > 0:
            cur_node, cur_dist = dist_list[0]
            del dist_list[0]

            for next_node in self.adj_list[cur_node]:
                if cur_dist < self.l0:
                    if self.dist_mat[node_s][next_node] is None:
                        self.dist_mat[node_s][next_node] = cur_dist+1
                        self.dist_mat[next_node][node_s] = cur_dist+1
                        dist_list.append((next_node, cur_dist+1))
                else:
                    if self.dist_mat[node_s][next_node] is None:
                        self.dist_mat[node_s][next_node] = -1
                        self.dist_mat[next_node][node_s] = -1
                        dist_list.append((next_node, cur_dist+1))

    # def cal_two_nodes_distance(self, node_s, node_t):
    #     if self.dist_mat[node_s][node_t] is not None:
    #         if self.dist_mat[node_s][node_t] < 0:
    #             return None
    #         return self.dist_mat[node_s][node_t]

    #     if node_s == node_t:
    #         return 0

    #     dist_list = [(node_s, 0)]
    #     res = None
    #     while len(dist_list) > 0:
    #         cur_node, cur_dist = dist_list[0]
    #         del dist_list[0]

    #         for next_node in self.adj_list[cur_node]:
    #             if next_node == node_t:
    #                 res = cur_dist+1
    #                 break
    #             if cur_dist < self.l0:
    #                 dist_list.append((next_node, cur_dist+1))
    #         if res is not None:
    #             break
        
    #     if res is None:
    #         self.dist_mat[node_s][node_t] = -1
    #         self.dist_mat[node_t][node_s] = -1
    #     else:
    #         self.dist_mat[node_s][node_t] = res
    #         self.dist_mat[node_t][node_s] = res
    #     return res

    def random_walk(self, node_s):
        res = set()
        for _ in range(self.l3):
            cur_node = node_s
            for _ in range(self.tao):
                if len(self.adj_list[cur_node]) == 0:
                    continue
                cur_node = random.choice(self.adj_list[cur_node])
                # distance = self.cal_two_nodes_distance(node_s, cur_node)
                distance = self.dist_mat[node_s][cur_node]
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
        source_node = self.train_node_list[idx]
        pos_nodes_list = self.random_walk(source_node)
        neg_nodes_list = self.neg_sample()

        labels = torch.tensor([1]*len(pos_nodes_list) + [-1]*len(neg_nodes_list))
        nodes_cont = torch.cat((self.raw_cont[pos_nodes_list], self.raw_cont[neg_nodes_list]))
        nodes_adj = torch.cat((self.adj_matrix[pos_nodes_list], self.adj_matrix[neg_nodes_list]))

        return nodes_adj, nodes_cont, labels, self.raw_cont[source_node], self.adj_matrix[source_node]

    def __len__(self):
        return len(self.train_node_list)
