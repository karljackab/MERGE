import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
import utils.Dataset as DS
import utils.Models as MD
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

dataset = 'cora'
BS = 128
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_weight = '1169_0.4879.pkl'

if dataset == 'cora':
    net_xn, node_xn = 2708, 1433

def fit_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    clf = LogisticRegression(max_iter=10000).fit(X_train, y_train)
    y_train_pred, y_test_pred = clf.predict(X_train), clf.predict(X_test)
    train_macro_f1, train_micro_f1, test_macro_f1, test_micro_f1 = \
        f1_score(y_train, y_train_pred, average='macro'), f1_score(y_train, y_train_pred, average='micro'), \
        f1_score(y_test, y_test_pred, average='macro'), f1_score(y_test, y_test_pred, average='micro')
    return train_macro_f1, train_micro_f1, test_macro_f1, test_micro_f1

if __name__ == "__main__":
    train_set = DS.Merge_Dataset(dataset, 0.1, (4, 9, 10 ,10))
    whole_set = DS.Merge_Whole_Dataset(train_set.adj_matrix, train_set.raw_cont, train_set.label_list)
    whole_data_loader = DataLoader(dataset=whole_set, batch_size=BS, num_workers=12)

    model = torch.load(f'weight/{model_weight}')

    tot_emb, tot_labels = [], []
    with torch.no_grad():
        for source_nodes_adj, source_nodes_cont, source_labels in tqdm(whole_data_loader):
            source_nodes_adj, source_nodes_cont = source_nodes_adj.to(device), source_nodes_cont.to(device)
            net_e, _, node_e, _ = model(source_nodes_adj, source_nodes_cont)
            # print(so_node_e.shape, so_net_e.shape)
            final_e = torch.cat((net_e, node_e), dim=1)
            tot_emb.append(final_e.cpu())
            tot_labels.append(source_labels)
        
        tot_emb = torch.cat(tot_emb).numpy()
        tot_labels = torch.cat(tot_labels).numpy()
        train_macro_f1, train_micro_f1, test_macro_f1, test_micro_f1 = fit_regression(tot_emb, tot_labels)
        print(f'Train macro f1: {train_macro_f1}, Train micro f1: {train_micro_f1}')
        print(f'Test macro f1: {test_macro_f1}, Test micro f1: {test_micro_f1}')
