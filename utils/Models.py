import torch
import torch.nn as nn

class AE(nn.Module):
    def __init__(self, net_xn, node_xn, final_en=128):
        super().__init__()
        self.net_xn = net_xn
        self.node_xn = node_xn
        self.final_en = final_en

        self.NetEncoder = nn.Sequential(
            nn.Linear(net_xn, 700),
            nn.BatchNorm1d(700),
            nn.ReLU(),
            nn.Linear(700, 500),
            # nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, 300),
            # nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Linear(300, final_en)
        )
        self.NetDecoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(final_en, 300),
            nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Linear(300, 500),
            # nn.BatchNorm1d(500),
            nn.ReLU(),
            nn.Linear(500, 700),
            # nn.BatchNorm1d(700),
            nn.ReLU(),
            nn.Linear(700, net_xn),
        )

        self.NodeEncoder = nn.Sequential(
            nn.Linear(node_xn, 450),
            nn.BatchNorm1d(450),
            nn.ReLU(),
            nn.Linear(450, 300),
            # nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Linear(300, 200),
            # nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, final_en)
        )
        self.NodeDecoder = nn.Sequential(
            nn.ReLU(),
            nn.Linear(final_en, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Linear(200, 300),
            # nn.BatchNorm1d(300),
            nn.ReLU(),
            nn.Linear(300, 450),
            # nn.BatchNorm1d(450),
            nn.ReLU(),
            nn.Linear(450, node_xn),
        )

    def forward_net(self, net_x):
        net_e = self.NetEncoder(net_x)
        net_x_rec = self.NetDecoder(net_e)
        return net_e, net_x_rec
    
    def forward_node(self, node_x):
        node_e = self.NodeEncoder(node_x)
        node_x_rec = self.NodeDecoder(node_e)
        return node_e, node_x_rec

    def forward(self, net_x, node_x):
        net_e, net_x_rec = self.forward_net(net_x)
        node_e, node_x_rec = self.forward_node(node_x)
        return net_e, net_x_rec, node_e, node_x_rec
