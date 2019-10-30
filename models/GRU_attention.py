import torch
import torch.nn as nn
import torch.nn.functional as F
from models.attention import Attention

class Model(nn.Module):
    def __init__(self, args, data):
        super(Model, self).__init__()
        # 数据的变量数（列数），论文中Table 1 里的 D
        self.variables = data.m
        # 模型隐藏状态中的特征数量，通过args.hidRNN指定
        self.hidR = args.hidRNN
        # 模型循环层的数量，可以理解为，如果该参数为2，则堆叠了两个LSTM。通过args.rnn_layers指定
        self.layers = args.rnn_layers
        # 模型使用的窗口尺寸，可以理解为取多长时间进行训练，在本例中取7 * 24 = 168，即一周。通过args.window指定
        self.window = args.window
        # 定义attention
        self.attention = Attention(seq_len=self.window, hidden_emb=self.hidR)

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
        # 在这里 * 2 是因为加入了attention，
        self.linear = nn.Linear(self.hidR * 2, self.variables)

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

        # 通过attention模块计算权重，将attention权重与隐藏层状态拼接
        output = output.permute(1, 0, 2).contiguous()
        c = self.attention(output)
        c = c.permute(1, 0, 2).contiguous()
        h = torch.cat((c, h), 2)

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
