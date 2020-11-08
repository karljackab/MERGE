import os
import numpy as np
from collections import defaultdict
import pickle
from tqdm import tqdm

dataset = 'microblogPCU'

adj_list = defaultdict(list)
l0 = 4

def cal_two_nodes_distance(dist_mat, node_s):
    dist_mat[node_s][node_s] = 0
    dist_list = [(node_s, 0)]

    while len(dist_list) > 0:
        cur_node, cur_dist = dist_list[0]
        del dist_list[0]

        for next_node in adj_list[cur_node]:
            if cur_dist < l0:
                if dist_mat[node_s][next_node] is None:
                    dist_mat[node_s][next_node] = cur_dist+1
                    dist_mat[next_node][node_s] = cur_dist+1
                    dist_list.append((next_node, cur_dist+1))
            else:
                if dist_mat[node_s][next_node] is None:
                    dist_mat[node_s][next_node] = -1
                    dist_mat[next_node][node_s] = -1
                    dist_list.append((next_node, cur_dist+1))

if __name__ == "__main__":
    if dataset == 'microblogPCU':
        NODE_NUM = 691
        dist_mat = []
        for i in range(NODE_NUM):
            adj_list[i].append(i)
            dist_mat.append([])
            for _ in range(NODE_NUM):
                dist_mat[-1].append(None)

        with open('dataset/microblogPCU/new_adj.csv', 'r') as f:
            for row in f.readlines():
                a, b = row.strip().split(',')
                a, b = int(a), int(b)
                adj_list[b].append(a)
                adj_list[a].append(b)

        for i in tqdm(range(NODE_NUM)):
            cal_two_nodes_distance(dist_mat, i)
        
        for i in range(NODE_NUM):
            for j in range(NODE_NUM):
                if dist_mat[i][j] is not None and dist_mat[i][j] < 0:
                    dist_mat[i][j] = None

        with open(f'dataset/{dataset}/dist_mat.pkl', 'wb') as f:
            pickle.dump(dist_mat, f)