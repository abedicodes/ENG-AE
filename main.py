import os
import pickle

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from tqdm import tqdm

from LSTMAE import lstmae
from utils import get_dataloader

torch.manual_seed(0)
use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")
print("Device being used:", device, flush=True)


num_epochs = 100
lr = .001

root = ''
csv = ''

phases = ['test']
data = {}

with open(os.path.join(root, 'test' + '.features'), 'rb') as File:
    data['test'] = pickle.load(File)


batch_size = 16
frequency = 5
features_vector = ['Behavioral', 'valence', 'arousal']
dataloader = {}
for phase in phases:
    dataloader[phase] = get_dataloader(batch_size,
                                       os.path.join(csv, 'test_' + '.csv'),
                                       data[phase],
                                       features_vector,
                                       frequency,
                                       labels=3)

dataset_sizes = {x: len(dataloader[x].dataset) for x in ['test']}


for inputs, labels, files in tqdm(dataloader['test']):
    print(inputs.shape)
    print(labels.shape)
    break

n_features = inputs.shape[2]
hidden_dim = 64
num_layers = 1

model = lstmae(n_features, hidden_dim, num_layers).cuda()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
scheduler = StepLR(optimizer, step_size=100, gamma=.1)
criterion = torch.nn.L1Loss(reduction='sum').cuda()


for epoch in range(num_epochs):

    for phase in ['train']:

        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = .0

        y_trues = np.empty([0])
        y_preds = np.empty([0])

        for inputs, labels, files in tqdm(dataloader['test']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            labels = labels.float()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                loss = criterion(outputs, inputs)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / dataset_sizes['test']

        print("[{}] Epoch: {}/{} Loss: {} LR: {}".format(
            phase, epoch + 1, num_epochs, epoch_loss, scheduler.get_last_lr()), flush=True)

        if phase == 'test':
            ind = y_trues.argsort()
            y_trues = y_trues[ind]
            y_preds = y_preds[ind]
            np.savetxt(os.path.join(root, 'y_trues.csv'), y_trues, delimiter=',')
            np.savetxt(os.path.join(root, 'y_preds.csv'), y_preds, delimiter=',')
