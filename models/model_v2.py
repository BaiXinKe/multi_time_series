import math
import torch
import torch.nn as nn
from torch.utils import data
from models.gcn_module import GCNBlock
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


class DecoderCell(nn.Module):
    def __init__(self, num_node, in_features, out_features) -> None:
        super(DecoderCell, self).__init__()
        self.reset_gate = GCNBlock(
            num_node=num_node, in_features=in_features + out_features, out_features=out_features)

        self.upate_gate = GCNBlock(
            num_node=num_node, in_features=in_features + out_features, out_features=out_features)

        self.candidate = GCNBlock(
            num_node=num_node, in_features=in_features + out_features, out_features=out_features)

    def forward(self, inputs, state, adj):
        """
        inputs: (batch, 1, num_node, in_features)
        state:  (batch, 1, num_node, out_feautes)
        """
        inputs, state = inputs.permute(0, 2, 1, 3), state.permute(0, 2, 1, 3)

        input_cat_state = torch.concat([inputs, state], dim=-1)
        Rt = torch.relu(self.reset_gate(input_cat_state, adj))
        Ut = torch.relu(self.upate_gate(input_cat_state, adj))
        Rt_dot_state = Rt * state
        X_cat_RS = torch.concat([inputs, Rt_dot_state], dim=-1)
        Ct = torch.tanh(self.candidate(X_cat_RS, adj))
        out = Ut * state + (1-Ut) * Ct
        return out


class Decoder(nn.Module):
    def __init__(self, n_layer, num_node, n_history, out_seqlen, in_features, out_features) -> None:
        super(Decoder, self).__init__()
        self.n_layer = n_layer
        self.num_node = num_node
        self.out_seqlen = out_seqlen
        self.out_features = out_features
        self.n_history = n_history
        self.layers = nn.ModuleList(
            [DecoderCell(num_node, in_features, out_features)])
        for i in range(1, self.n_layer):
            self.layers.append(out_features, out_features)

    def forward(self, inputs, adj):
        """
        inputs: (batch_size, 1, num_node, in_features)
        """
        state = torch.zeros(
            inputs.shape[0], inputs.shape[1], self.out_features).to(inputs.device)

        state = self.layers[0](inputs, state, adj)

        for i in range(self.n_layer):
            state = self.layers[i]()


if __name__ == '__main__':
    net = DecoderCell(307, 32, 32)
    adj = torch.randn(307, 307)
    data = torch.randn(64,  12, 307, 32)
    state = torch.randn(64,  12, 307, 32)
    res = net(data, state, adj)
    print(res.shape)
