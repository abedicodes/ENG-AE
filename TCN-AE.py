import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import random
import torch.utils.data as data_utils
import matplotlib.pyplot as plt
from torchvision import models
from torch.nn.utils import weight_norm
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve, r2_score, confusion_matrix, accuracy_score, \
    precision_recall_curve, auc
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import pandas as pd


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2, last_layer=False):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.with_relu = not last_layer

        if (self.with_relu):
            self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                     self.conv2, self.chomp2, self.relu2, self.dropout2)
        else:
            self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                     self.conv2, self.chomp2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        # self.conv1.weight.data.normal_(0, 0.01)
        # self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            # self.downsample.weight.data.normal_(0, 0.01)
            nn.init.xavier_uniform_(self.downsample.weight)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        if (self.with_relu):
            return self.relu(out + res)
        else:
            return out + res


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, dilation_increase=True):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i if (dilation_increase) else 2 ** (num_levels - 1 - i)
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            if dilation_increase == False and i == num_levels - 1:
                layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                         dilation=dilation_size, padding=(kernel_size - 1) * dilation_size,
                                         dropout=dropout, last_layer=True)]
            else:
                layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1,
                                         dilation=dilation_size, padding=(kernel_size - 1) * dilation_size,
                                         dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)

    def forward(self, inputs):
        """Inputs have to have dimension (batch_size, num_features, seq_length)"""
        y1 = self.tcn(inputs)  # input should have dimension (batch_size, num_features, seq_length)
        o = self.linear(y1[:, :, -1])
        return F.log_softmax(o, dim=1)


class AE_TCN(nn.Module):
    def __init__(self, n_inputs, n_channels, kernel_size=3, dropout=0.2):
        super(AE_TCN, self).__init__()

        self.en_TCN = TemporalConvNet(n_inputs, n_channels, kernel_size, dropout)
        self.de_TCN = TemporalConvNet(n_inputs, n_channels, kernel_size, dropout)

        self.en_conv = nn.Conv1d(in_channels=n_channels[-1], out_channels=60, kernel_size=1)
        self.de_conv = nn.Conv1d(in_channels=n_channels[-1], out_channels=60, kernel_size=1)

        # decode_num_channels = n_channels[:-1]
        # decode_num_channels.reverse()
        # decode_num_channels += [n_inputs]
        # self.de_TCN_ = TemporalConvNet(n_channels[-1], decode_num_channels, kernel_size, dropout,
        #                                dilation_increase=False)
        #
        # self.linear = nn.Linear(in_features=n_channels[-1], out_features=1)

    def forward(self, x):
        x = self.en_TCN(x)
        x = self.en_conv(x)

        latent = F.avg_pool1d(x, 2)
        out = F.interpolate(latent, size=x.shape[2])

        out = self.de_TCN(out)
        out = self.de_conv(out)

        # x = self.de_conv(x)
        # out = self.de_TCN_(x)
        # out = self.de_TCN(x)
        # out = out.permute((0, 2, 1))
        # out = self.linear(out)
        # out = out.permute((0, 2, 1))

        return out


random_state = 24
torch.manual_seed(random_state)
use_cuda = False
device = torch.device("cuda" if use_cuda else "cpu")

features_train, targets_train = torch.load('features_emotiw_train.pt')
# features_test, targets_test = torch.load('features_emotiw_test.pt')
features_test, targets_test, four_level_targets_test = torch.load('features_emotiw_test_0_60.pt')

# features_train, targets_train = torch.load('features_emotiw_train.pt')
# features_test, targets_test = torch.load('features_emotiw_test.pt')

features_train_normal = features_train[targets_train == 0]
targets_train_normal = targets_train[targets_train == 0]

features_test_normal = features_test[targets_test == 0]
targets_test_normal = targets_test[targets_test == 0]

features_train_anomalous = features_train[targets_train == 1]
targets_train_anomalous = targets_train[targets_train == 1]

features_test_anomalous = features_test[targets_test == 1]
targets_test_anomalous = targets_test[targets_test == 1]
# --------------------------------------------------------------------------------
features_test = torch.cat([features_test_normal, features_test_anomalous])
targets_test = torch.cat([torch.Tensor(targets_test_normal), torch.Tensor(targets_test_anomalous)])

train_dataset = TensorDataset(torch.Tensor(features_train_normal), torch.Tensor(targets_train_normal))
train_loader = DataLoader(train_dataset, batch_size=16)

test_dataset = TensorDataset(features_test, targets_test)
test_loader = DataLoader(test_dataset, batch_size=1)

input_channels = features_train.shape[2]
seq_length = features_train.shape[1]

nhid = 25
levels = 8
channel_sizes = [nhid] * levels
kernel_size = 7
dropout = .05
# model = TCN(input_channels, n_classes, channel_sizes, kernel_size=kernel_size, dropout=dropout)
model = AE_TCN(input_channels, channel_sizes, kernel_size=kernel_size, dropout=dropout)

epochs = 250
lr = 1e-2
gamma = .7
optimizer = optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
loss_func = nn.MSELoss()

# test dimensions
# n_inputs = input_channels
# n_channels = channel_sizes
# en_TCN = TemporalConvNet(n_inputs, n_channels, kernel_size, dropout)
# en_conv = nn.Conv1d(in_channels=n_channels[-1], out_channels=60, kernel_size=1)
# de_conv = nn.Conv1d(in_channels=n_channels[-1], out_channels=60, kernel_size=1)
# # de_conv = nn.Conv1d(25, 10, 1)
# linear = nn.Linear(25, out_features=1)
# de_TCN = TemporalConvNet(n_inputs, n_channels, kernel_size, dropout)
# #
# x = data
# x = en_TCN(x)
# x = en_conv(x)
# latent = F.avg_pool1d(x, 2)
# x = F.interpolate(latent, size=x.shape[2])
# out = de_TCN(x)
# out = out.permute((0, 2, 1))
# out = linear(out)
# out = linear(out)
# out = out.permute((0, 2, 1))
#
# out = de_conv(out)

roc_auc = []
roc_pr = []

for epoch in range(1, epochs + 1):

    for batch_idx, (data, target) in enumerate(train_loader):

        data, target = data.to(device), target.to(device)

        data = data.view(-1, input_channels, seq_length)

        # break

        optimizer.zero_grad()
        output = model(data)
        loss = loss_func(output, data)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader), loss.item()), flush=True)

# torch.save(model.state_dict(), '/home/ali/PycharmProjects/BoW/model.pth')
# model.load_state_dict(torch.load('/home/ali/PycharmProjects/BoW/model.pth'))

# test
model.eval()
losses = []
targets = []
# model.eval()
test_loss = 0
correct = 0
with torch.no_grad():
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        data = data.view(-1, input_channels, seq_length)
        output = model(data)
        loss = loss_func(output, data)
        losses.append(loss.item())
        targets.append(target.item())
        # print(loss.item(), '\t', target.item())
