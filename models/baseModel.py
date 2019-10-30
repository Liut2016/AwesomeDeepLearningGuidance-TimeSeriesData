import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import Attention

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        '''
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

        '''
        # 数据的变量数（列数），论文中Table 1 里的 D
        self.variables = data.m
        # 模型隐藏状态中的特征数量，通过args.hidRNN指定
        self.hidR = args.hidRNN
        # 模型循环层的数量，可以理解为，如果该参数为2，则堆叠了两个LSTM。通过args.rnn_layers指定
        self.layers = args.rnn_layers

        # 注意： 要把不用的层先注释掉！
        # 例如，如果想跑 GRU，就把 self.rnn 和 self.lstm 注释掉，否则会报错

        # 定义 RNN 模型
        # 与 GRU 模型参数相同
        '''
        self.rnn = nn.RNN(
            input_size = self.variables,
            hidden_size = self.hidR,
            num_layers = self.layers,
            bidirectional = False
        )
        '''

        # 定义 LSTM 模型
        # pytorch中模型定义见 https://pytorch.apachecn.org/docs/1.2/nn.html
        #
        # Parameters:
        #       input_size – The number of expected features in the input x
        #       hidden_size – The number of features in the hidden state h
        #       num_layers – Number of recurrent layers.E.g., setting num_layers = 2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results.Default: 1
        #       bias – If False, then the layer does not use bias weights b_ih and b_hh.Default: True
        #       batch_first – If True, then the input and output tensors are provided as (batch, seq, feature).Default: False
        #       dropout – If non - zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout.Default: 0
        #       bidirectional – If True, becomes a bidirectional LSTM.Default: False
        #
        # 输入：input, (h_0, c_0)
        #      input：(seq_len, batch, input_size)的三维张量
        #      h_0, c_0 (num_layers * num_directions, batch, hidden_size)分别表示模型的初始隐藏状态和初始细胞状态，如果不输入，则默认为0
        # 输出：output, (h_n, c_n)
        #      output：(seq_len, batch, num_directions * hidden_size)的三维张量
        #      h_0, c_0 (num_layers * num_directions, batch, hidden_size)分别表示模型最后一个细胞的隐藏状态和细胞状态
        #      一般使用
        #
        '''
        self.lstm = nn.LSTM(
            input_size = self.variables,
            hidden_size = self.hidR,
            num_layers = self.layers,
            bidirectional = False
        )
        '''

        # 定义 GRU 模型
        # pytorch中模型定义见 https://pytorch.apachecn.org/docs/1.2/nn.html
        #
        # Parameters:
        #       input_size – The number of expected features in the input x
        #       hidden_size – The number of features in the hidden state h
        #       num_layers – Number of recurrent layers.E.g., setting num_layers = 2 would mean stacking two LSTMs together to form a stacked LSTM, with the second LSTM taking in outputs of the first LSTM and computing the final results.Default: 1
        #       bias – If False, then the layer does not use bias weights b_ih and b_hh.Default: True
        #       batch_first – If True, then the input and output tensors are provided as (batch, seq, feature).Default: False
        #       dropout – If non - zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer, with dropout probability equal to dropout.Default: 0
        #       bidirectional – If True, becomes a bidirectional LSTM.Default: False
        #
        # 输入：input, h_0
        #      input：(seq_len, batch, input_size)的三维张量
        #      h_0 (num_layers * num_directions, batch, hidden_size)表示模型的初始隐藏状态，如果不输入，则默认为0
        # 输出：output, h_n
        #      output：(seq_len, batch, num_directions * hidden_size)的三维张量
        #      h_0 (num_layers * num_directions, batch, hidden_size)表示模型最后一个细胞的隐藏状态
        #      一般使用
        #

        self.gru = nn.GRU(
            input_size = self.variables,
            hidden_size = self.hidR,
            num_layers = self.layers,
            bidirectional = False
        )




        # 定义全连接层
        # 输入：RNN模型的隐藏状态
        # 输出：预测的一条时序数据的长度（变量数量），论文Table1中的D
        self.linear = nn.Linear(self.hidR, self.variables)

        # dropout模块，通过args.dropout参数指定丢弃率
        self.dropout = nn.Dropout(p=args.dropout)
        # 通过args.output_fun参数选择结果的激活函数
        self.output = None
        if (args.output_fun == 'sigmoid'):
            self.output = torch.sigmoid
        if (args.output_fun == 'tanh'):
            self.output = torch.tanh

    def forward(self, x):
        # x为(batch, seq_len, input_size)，需要将其转化为模型输入对应的格式
        r = x.permute(1,0,2).contiguous()
        # output：(seq_len, batch, num_directions * hidden_size)的三维张量，包含所有时刻隐藏状态的值
        # (h_n, c_n)元组表示最后一层的隐藏状态和细胞状态

        # 使用 RNN 模型
        #output, h = self.rnn(r)
        # 使用 LSTM 模型
        #output, (h, c) = self.lstm(r)
        # 使用 GRU 模型
        output, h = self.gru(r)


        # 运行这几条语句可以得出，output的最后一行与和h相同
        # 如果bidirectional=False, output1尺寸为(128, 100), output2尺寸为(1, 128, 100), output2 = h
        # 如果bidirectional=True, output1尺寸为(128, 200), output2尺寸为(1, 128, 200), h 尺寸为(2, 128, 100)
        '''
        print("output1", output[-1,:,:], output[-1,:,:].size())
        print("output2", output[-1:,:,:], output[-1:,:,:].size())
        print("h", h, h.size())
        '''

        # h[-1:, : , :]表示取最后一个 LSTM 的 h_n
        # h_n为(num_layers * num_directions, batch, hidden_size)大小的张量
        # torch.squeeze(a, b)表示从a中去掉b这个维度，在这里即把 num_layers * num_directions这个维度去掉
        # 进行dropout操作
        r = self.dropout(torch.squeeze(h[-1:,:,:], 0))
        # 获取到隐藏层状态后，进行下一步分类或回归任务
        # 在这里是将隐藏层状态放入到全连接层中，输出大小为要预测的时间序列的维度（变量个数）
        out = self.linear(r)
        #print("out_before", out)
        # 预测结果为什么还要激活？
        if self.output is not None:
            out=self.output(out)
        #print("out_after", out)
        return out
