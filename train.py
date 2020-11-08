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
import os

EPOCH = 10000
dataset = 'citeseer'
train_data_ratio = 1
l0, l1, tao, l3 = 4, 9, 10, 10
BS = 256
LR = 0.05
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
alpha, beta, gamma = 1e-4, 1e-3, 1e-5
suffix = 'with_first_bn'
save_model_pth = f'weight/{dataset}_{BS}_{train_data_ratio}_{LR}'
save_log_pth = f'log/{dataset}_{BS}_{train_data_ratio}_{LR}'

if suffix is not None:
    save_model_pth = f'{save_model_pth}_{suffix}'
    save_log_pth = f'{save_log_pth}_{suffix}'

if not os.path.exists(save_log_pth):
    os.makedirs(save_log_pth)
if not os.path.exists(save_model_pth):
    os.makedirs(save_model_pth)

print(f'Suffix: {suffix}')
if dataset == 'cora':
    if suffix == 'pca':
        net_xn, node_xn = 1150, 650
    else:
        net_xn, node_xn = 2708, 1433
elif dataset == 'citeseer':
    if suffix == 'pca':
        net_xn, node_xn = 3000, 1300
    else:
        net_xn, node_xn = 3312, 3703
elif dataset == 'microblogPCU':
    if suffix == 'pca':
        net_xn, node_xn = 691, 3836
    else:
        net_xn, node_xn = 691, 3836

def merge_loss(so_net, so_node, so_net_e, so_net_rec, so_node_e, so_node_rec, net_e, node_e, labels, source_mapping_idx):
    o2 = alpha*nn.MSELoss(reduction='sum')(so_net_rec, so_net)
    o3 = beta*nn.MSELoss(reduction='sum')(so_node_rec, so_node)

    o1 = 0
    for idx in range(len(node_e)):
        o1 += F.logsigmoid(torch.matmul(node_e[idx], so_net_e[source_mapping_idx[idx]]) * labels[idx]) 
    o1 /= len(node_e)
    return -o1+o2+o3

def fit_regression(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    clf = LogisticRegression(max_iter=10000).fit(X_train, y_train)
    y_train_pred, y_test_pred = clf.predict(X_train), clf.predict(X_test)
    train_macro_f1, train_micro_f1, test_macro_f1, test_micro_f1 = \
        f1_score(y_train, y_train_pred, average='macro'), f1_score(y_train, y_train_pred, average='micro'), \
        f1_score(y_test, y_test_pred, average='macro'), f1_score(y_test, y_test_pred, average='micro')
    return train_macro_f1, train_micro_f1, test_macro_f1, test_micro_f1

if __name__ == "__main__":
    model = MD.AE(net_xn, node_xn).to(device)
    train_set = DS.Merge_Dataset(dataset, train_data_ratio, (l0, l1, tao ,l3), suffix)
    whole_set = DS.Merge_Whole_Dataset(train_set.adj_matrix, train_set.raw_cont, train_set.label_list)
    train_loader = DataLoader(dataset=train_set, batch_size=BS, shuffle=True, num_workers=12, collate_fn=DS.Merge_collfn) ## , drop_last=True
    whole_data_loader = DataLoader(dataset=whole_set, batch_size=BS, num_workers=12)
    opt = optim.SGD(model.parameters(), lr=LR, momentum=0.9, weight_decay=gamma)
    best_loss = 10
    for epoch in range(EPOCH):
        print(f'Epoch {epoch} start')
        model.train()
        tot_loss, cnt = 0, 0
        
        for nodes_adj, nodes_cont, labels, source_nodes_cont, source_nodes_adj, source_mapping_idx\
                 in tqdm(train_loader):
            nodes_adj, nodes_cont = nodes_adj.to(device), nodes_cont.to(device)
            source_nodes_adj, source_nodes_cont = source_nodes_adj.to(device), source_nodes_cont.to(device)
            
            net_e, _, node_e, _ = model(nodes_adj, nodes_cont)
            so_net_e, so_net_rec, so_node_e, so_node_rec = model(source_nodes_adj, source_nodes_cont)
            loss = merge_loss(source_nodes_adj, source_nodes_cont, so_net_e, so_net_rec, so_node_e, so_node_rec, net_e, node_e, labels, source_mapping_idx)

            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_loss += loss.detach().cpu()
            cnt += 1

        cur_loss = tot_loss/cnt
        print(f'loss: {cur_loss}')
        if best_loss > cur_loss and cur_loss < 5.5:
            torch.save(model, f'{save_model_pth}/{epoch}_{cur_loss:.4f}.pkl')
            best_loss = cur_loss

        tot_emb, tot_labels = [], []
        model.eval()
        print('start eval')
        with torch.no_grad():
            for source_nodes_adj, source_nodes_cont, source_labels in tqdm(whole_data_loader):
                source_nodes_adj, source_nodes_cont = source_nodes_adj.to(device), source_nodes_cont.to(device)
                net_e, _, node_e, _ = model(source_nodes_adj, source_nodes_cont)
                final_e = torch.cat((net_e, node_e), dim=1)
                tot_emb.append(final_e.cpu())
                tot_labels.append(source_labels)

        tot_emb = torch.cat(tot_emb).numpy()
        tot_labels = torch.cat(tot_labels).numpy()
        train_macro_f1, train_micro_f1, test_macro_f1, test_micro_f1 = fit_regression(tot_emb, tot_labels)
        print(f'Train macro f1: {train_macro_f1}, Train micro f1: {train_micro_f1}')
        print(f'Test macro f1: {test_macro_f1}, Test micro f1: {test_micro_f1}')

        with open(f'{save_log_pth}/loss', 'a') as f_loss, \
                open(f'{save_log_pth}/train_macro_f1', 'a') as f_train_macro_f1, open(f'{save_log_pth}/train_micro_f1', 'a') as f_train_micro_f1,\
                open(f'{save_log_pth}/test_macro_f1', 'a') as f_test_macro_f1, open(f'{save_log_pth}/test_micro_f1', 'a') as f_test_micro_f1:
            f_loss.write(f'{cur_loss}\n')
            f_train_macro_f1.write(f'{train_macro_f1}\n')
            f_train_micro_f1.write(f'{train_micro_f1}\n')
            f_test_macro_f1.write(f'{test_macro_f1}\n')
            f_test_micro_f1.write(f'{test_micro_f1}\n')
