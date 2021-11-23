import math
import torch
from torch._C import dtype
import torch.nn as nn
import numpy as np
from torch.nn.functional import log_softmax
from torch.nn.modules.container import ModuleList
from torch.utils import data
from datareader.with_preweek_dataset import Split_last_week_filter


class TCNLayer(nn.Module):
    def __init__(self, in_features, out_features, kernel=3):
        super(TCNLayer, self).__init__()
        self.conv1 = nn.Conv2d(in_features, out_features,
                               kernel_size=(1, kernel))
        self.conv2 = nn.Conv2d(in_features, out_features,
                               kernel_size=(1, kernel))
        self.conv3 = nn.Conv2d(in_features, out_features,
                               kernel_size=(1, kernel))

    def forward(self, inputs):
        """
        param inputs: (batch_size, timestamp, num_node, in_features)
        return: (batch_size, timestamp - 2, num_node, out_features)
        """
        inputs = inputs.permute(0, 3, 2, 1)  # (btnf->bfnt)
        out = self.conv1(inputs) + torch.sigmoid(self.conv2(inputs))
        out = torch.relu(out + self.conv3(inputs))
        out = out.permute(0, 3, 2, 1)
        return out


class TCN(nn.Module):
    def __init__(self, n_history, in_features, mid_features) -> None:
        super(TCN, self).__init__()
        # odd time seriers: number of layer is n_hitory // 2
        # even time seriers: number of layer is n_history//2-1 + a single conv layer.
        # -> Aggregate information from time seriers to one unit
        assert(n_history >= 3)
        self.is_even = False if n_history % 2 != 0 else True

        self.n_layers = n_history // \
            2 if n_history % 2 != 0 else (n_history // 2 - 1)

        self.tcn_layers = nn.ModuleList([TCNLayer(in_features, mid_features)])
        for i in range(self.n_layers - 1):
            self.tcn_layers.append(TCNLayer(mid_features, mid_features))

        if self.is_even:
            self.tcn_layers.append(
                TCNLayer(mid_features, mid_features, kernel=2))

        self.upsample = None if in_features == mid_features else nn.Linear(
            in_features, mid_features)

    def forward(self, inputs):
        out = self.tcn_layers[0](inputs)
        if self.upsample:
            inputs = self.upsample(inputs)

        out = out + inputs[:, 2:, ...]

        for i in range(1, self.n_layers):
            out = self.tcn_layers[i](out) + out[:, 2:, ...]

        if self.is_even:
            out = self.tcn_layers[-1](out) + out[:, -1, :, :].unsqueeze(1)

        return out


class GCNCell(nn.Module):
    def __init__(self, in_features, out_features) -> None:
        super(GCNCell, self).__init__()
        self.weight = nn.Parameter(
            torch.FloatTensor(in_features, out_features))
        self.bais = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameter()

    def reset_parameter(self):
        stdv = 1. / math.sqrt(self.weight.shape[1])
        self.weight.data.uniform_(-stdv, stdv)
        self.bais.data.zero_()

    def forward(self, inputs, adj):
        """
        inputs: (batch_size, num_node, num_features)
        adj: (num_node, num_node)
        """
        lfs = torch.einsum('ij,jbf->bif', adj, inputs.permute(1, 0, 2))
        print(self.weight.shape)
        print(lfs.shape)
        result = torch.relu(torch.matmul(lfs, self.weight) + self.bais)
        return result


class DecoderCell(nn.Module):
    def __init__(self, num_node, in_features, out_features) -> None:
        super(DecoderCell, self).__init__()
        self.reset_gate = GCNCell(
            in_features + out_features, out_features)
        self.br = nn.Parameter(torch.zeros(num_node, out_features))
        self.update_gate = GCNCell(
            in_features + out_features, out_features)
        self.bu = nn.Parameter(torch.zeros(num_node, out_features))
        self.candidate = GCNCell(
            in_features + out_features, out_features)
        self.bc = nn.Parameter(torch.zeros(num_node, out_features))

    def forward(self, inputs, state, adj):
        """
        inputs: (batch_size, num_node, in_features)
        state : (batch_size, num_node, out_features)
        return: (batch_size, num_node, out_features)
        """
        concat_xs = torch.concat([inputs, state], dim=-1)
        Rt = torch.relu(self.reset_gate(concat_xs, adj) + self.br)
        Ut = torch.relu(self.update_gate(concat_xs, adj) + self.bu)
        Rt_dot_state = Rt * state
        cat_xrts = torch.concat([inputs, Rt_dot_state], dim=-1)
        Ct = torch.tanh(self.candidate(cat_xrts, adj) + self.bc)
        out = Ut * state + (1-Ut) * Ct
        return out


class Decoder(nn.Module):
    def __init__(self, n_layer, num_node, n_predict, adj, in_features, mid_features, out_features) -> None:
        super(Decoder, self).__init__()

        self.n_layer = n_layer
        self.num_node = num_node
        self.n_predict = n_predict

        self.mid_features = mid_features

        self.adj = nn.Parameter(torch.from_numpy(adj))

        self.layers = ModuleList(
            [DecoderCell(num_node, in_features, mid_features)])
        for i in range(1, self.n_layer):
            self.layers.append(DecoderCell(
                num_node, mid_features, mid_features))

        self.linear = nn.Parameter(
            torch.FloatTensor(num_node, mid_features, out_features))

        self.reset_parameter()

    def reset_parameter(self):
        stdv = 1. / math.sqrt(self.linear.shape[1])
        self.linear.data.uniform_(-stdv, stdv)

    def forward(self, inputs, preweek_pre):
        """
        inputs: (batch_size, 1, num_node, in_features)
        """
        inputs = inputs[:, 0, :, :]
        state = torch.zeros(
            inputs.shape[0], inputs.shape[1], self.mid_features)
        hidden_state = [state]
        for i in range(self.n_predict - 1):
            state = hidden_state[-1]
            for j in range(self.n_layer):
                state = self.layers[j](
                    preweek_pre[:, i, :, :], state, self.adj)
                # !!! state = self.layers[j]( ,state, self.adj)
                # 这里就是现在要考虑的重点，如何就组织上一周的数据和本周预测的数据
                # 从NLP来开，输入到decoder的是目标序列，因此，我们输入可以使用这个
        result = torch.stack(hidden_state, dim=1)
        print(result.shape)


if __name__ == '__main__':
    adj = np.random.randn(307, 307).astype(np.float32)
    net = Decoder(n_layer=2, num_node=307, n_predict=12, adj=adj,
                  in_features=1, mid_features=32, out_features=1)
    data = torch.randn(64, 12, 307, 1)
    preweek_pre = torch.randn(64, 12, 307, 1)

    res = net(data, preweek_pre)

    print(res.shape)
