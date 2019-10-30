import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model,self).__init__()
        self.hidC = args.hidCNN
        self.variables = data.m
        self.P = args.window
        self.m = data.m
        self.Ck = args.CNN_kernel
        self.dropout = nn.Dropout(p=args.dropout)

        self.conv = nn.Conv2d(1, self.hidC, kernel_size=(self.Ck, self.m))

        # 这里的 16300 其实应该根据上面的参数计算得到，但是由于计算公式较为复杂，故直接把参数写死了
        self.linear = nn.Linear(16300, self.m)

        # 通过args.output_fun参数选择结果的激活函数
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = torch.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = torch.tanh


    def forward(self, x):
        #x = self.features(x)
        #x = x.view(x.size(0), 256 * 6 * 6)
        #x = self.classifier(x)
        x = x.view(-1, 1, self.P, self.m);
        x = F.relu(self.conv(x))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)

        if self.output is not None:
            out = self.output(x)
        return out
