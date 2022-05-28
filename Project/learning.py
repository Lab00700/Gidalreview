import sqlite3
import pandas as pd
import numpy as np
import os
import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

import warnings
warnings.filterwarnings('ignore')

class Learn(nn.Module):
    def __init__(self,word_size,input_size,hidden_size,n_layer=1,dropout_p=0.2):
        super(Learn,self).__init__()
        self.embed = nn.Embedding(word_size+1,input_size)
        self.rnn = []
        self.dropout = nn.Dropout(dropout_p)
        self.out1 = nn.RNN(hidden_size, 100)
        self.out2 = nn.RNN(100, 20)
        self.out3 = nn.RNN(20, 5)
        self.out4 = nn.RNN(5, 1)
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.n_layer=n_layer

    def forward(self,x):
        x = self.embed(x)

        for i in range(self.n_layer):
            if len(self.rnn)!=self.n_layer:
                rnn=nn.RNN(self.input_size, self.hidden_size,batch_first=True)
                self.rnn.append(rnn)
            else:
                rnn=self.rnn[i]
            x = rnn(x)
            x = self.dropout(x[0])
        x = self.out1(x)
        x = self.out2(x[0])
        x = self.out3(x[0])
        x = self.out4(x[0])
        return x[0]

    def start_learn(self, model,x,y):
        optimizer = torch.optim.Adam(model.parameters())

        x = torch.tensor(x)
        y = torch.tensor(y)

        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=False)

        model.train()
        for x1, y1 in loader:
            optimizer.zero_grad()

            logit = model(x1)
            loss = F.cross_entropy(logit, y1).to('cuda')
            loss.backward()
            optimizer.step()

        corrects, total_loss = 0, 0
        model.eval()
        for x1, y1 in loader:
            logit = model(x1)
            loss = F.cross_entropy(logit, y1, reduction='sum').to('cuda')
            total_loss += loss.item()
            corrects += (logit.max(1)[1].view(y1.size()).data == y1.data).sum()

        size = len(y)
        avg_loss = total_loss / size
        avg_accuracy = 100.0 * corrects / size
        return avg_loss, avg_accuracy
