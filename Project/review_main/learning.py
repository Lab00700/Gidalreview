import sqlite3
import pandas as pd
import numpy as np
import os
import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models

import warnings
warnings.filterwarnings('ignore')

class Net(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=True):
        super(Net, self).__init__()
        self.rnn_layer = nn.LSTM(input_size, hidden_size,batch_first=batch_first)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x):
        output, hidden = self.rnn_layer(x)
        output = self.linear(output)
        return output

class Embed(nn.Module):
    def __init__(self, vocab_size, input_size):
        super(Embed, self).__init__()
        self.embedding_layer = nn.Embedding(num_embeddings=vocab_size, # 워드 임베딩
                                            embedding_dim=input_size)
    def forward(self, x):
        output = self.embedding_layer(x)
        return output

def train(model, device, train_loader, optimizer, loss_function, epoch):
    try:
        model.load_state_dict(torch.load('DeepLearn.pth'))
    except:
        print("Model Error")
    model.train().to(device)
    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device)

        target = target.to(device)
        optimizer.zero_grad()
        output = model(data).to(device)
        loss = loss_function(output, target)
        loss.backward(retain_graph=True)
        optimizer.step()
        if batch_idx % 1000 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    torch.save(model.state_dict(), 'DeepLearn.pth')

def test(model, device, test_loader, loss_function):
    model.load_state_dict(torch.load('DeepLearn.pth'))

    model.eval().to(device)
    test_loss = 0
    correct = 0
    total_pred=[]
    with torch.no_grad():
        for data, target in test_loader:
            data = data.to(device)
            target = target.to(device)
            output = model(data).to(device)
            test_loss += loss_function(output, target).to(device)  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            if pred.item()<5:
                total_pred.append(pred.item())
            elif pred.item()==0:
                total_pred.append(1)
            else:
                total_pred.append(5)
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return total_pred

def start_learn(x, y, word_size, input_size,mode="train"):
    use_cuda = True
    device = torch.device("cuda" if use_cuda else "cpu")

    epochs = 10

    model = Net(input_size, input_size*2)

    optimizer = optim.Adam(model.parameters(), lr=0.001)

    x = torch.tensor(x).long()
    y = torch.tensor(y).long()

    embed = Embed(word_size+1, input_size)
    x=embed(x)

    loader = TensorDataset(x, y)
    loader = DataLoader(loader, batch_size=1, drop_last=False)

    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    loss_function = nn.CrossEntropyLoss()
    if mode=="train":
        for epoch in range(epochs):
            train(model, device, loader, optimizer, loss_function, epoch)
            test(model, device, loader, loss_function)
    else:
        return test(model, device, loader, loss_function)
