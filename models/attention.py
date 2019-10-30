# Attention Network

import torch
import torch.nn as nn
from torch.autograd import Variable

class Attention(nn.Module):
	'''
	Class to define the attention network component (attention is computed over the entire sequence length so
	that the information from each time step can be used in the final prediction and not just the final time step)
	The network is a 3 layer MLP architecture followed by a softmax
	Arguments:
		seq_len : maximum sequence length (this is required to predefine the dimensions of the network)
		hidden_emb : same as the hidden dimension of the lstm network
	Returns:
		None
	'''
	def __init__(self, seq_len=18, hidden_emb=1024):
		super(Attention, self).__init__()

		self.seq_len = seq_len
		self.hidden_emb = hidden_emb
		self.mlp1_units = 512
		self.mlp2_units = 128

		self.fc = nn.Sequential(
            nn.Linear(self.seq_len*self.hidden_emb, self.mlp1_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.mlp1_units, self.mlp2_units),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(self.mlp2_units, self.seq_len),
            nn.ReLU(inplace=True),
            )

		self.softmax = nn.Softmax(dim=1)

	'''
	Computes the attention on the lstm outputs from each time step in the sequence
	Arguments:
		lstm_emd : lstm embedding from each time step in the sequence
	Returns:
		attn_feature_map : embedding computed after applying attention to the lstm embedding of the entire sequence
	'''
	def forward(self, lstm_emd):

		batch_size = lstm_emd.shape[0]
		lstm_emd = lstm_emd.contiguous()
		lstm_flattened = lstm_emd.view(batch_size, -1) # to pass it to the MLP architecture

		attn = self.fc(lstm_flattened) # attention over the sequence length
		alpha = self.softmax(attn) # gives the probability values for the time steps in the sequence (weights to each time step)

		# [tensor]使其变成了[tensor]，[tensor] * num 表示将tensor复制 num 个，然后拼接在一起
		#alpha = torch.stack([alpha]*self.mlp2_units, dim=2) # stack it across the lstm embedding dimesion
		alpha = torch.stack([alpha] * self.hidden_emb, dim=2)
		attn_feature_map = lstm_emd * alpha # gives attention weighted lstm embedding
		attn_feature_map = torch.sum(attn_feature_map, dim=1, keepdim=True) # computes the weighted sum
		return attn_feature_map

# testing code
if __name__ == '__main__':
	net = Attention()

	lstm_emd = Variable(torch.Tensor(4,18,1024))

	out = net(lstm_emd)