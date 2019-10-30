__author__ = "Guan Song Wang"

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import Attention

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        self.window = args.window
        self.variables = data.m
        self.hw=args.highway_window
        self.activate1=F.relu
        self.hidR=args.hidRNN
        self.rnn1=nn.LSTM(self.variables,self.hidR,num_layers=args.rnn_layers,bidirectional=False)
        self.linear1 = nn.Linear(self.hidR, self.variables)
        # self.linear1=nn.Linear(1280,100)
        # self.out=nn.Linear(100,self.variables)
        if (self.hw > 0):
            self.highway = nn.Linear(self.hw, 1)
        print(self.hidR)
        print(self.window)
        #self.attention = Attention(hidden_emb=self.hidR, seq_len=self.window) # attention module
        self.attention = Attention(hidden_emb=self.hidR, seq_len=128) # attention module


        self.dropout = nn.Dropout(p=args.dropout)
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = F.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = F.tanh

    def forward(self, x):
        r= x.permute(1,0,2).contiguous()
        _,r=self.rnn1(r)
        #r = self.attention(r)
        #print(r)
        #print(r[0][-1:,:,:])
        #print(torch.squeeze(r[0][-1:,:,:], 0))
        #r = self.dropout(torch.squeeze(r[-1:, :, :], 0))
        # 针对LSTM
        r=self.dropout(torch.squeeze(r[0][-1:,:,:], 0))
        out = self.linear1(r)


        if (self.hw > 0):

            z = x[:, -self.hw:, :]
            z = z.permute(0, 2, 1).contiguous().view(-1, self.hw)
            z = self.highway(z)
            z = z.view(-1, self.variables)
            out = out + z
        if self.output is not None:
            out=self.output(out)
        return out
