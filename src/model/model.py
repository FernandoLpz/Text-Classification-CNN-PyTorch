import torch
import torch.nn as nn
import torch.nn.functional as F

class TextClassifier(nn.ModuleList):

	def __init__(self, params):
		super(TextClassifier, self).__init__()
		
		self.batch_size = params.batch_size
		self.seq_len = params.seq_len
		self.vocab_size = params.vocab_size
		self.embedding_size = params.embedding_size
		self.out_size = params.out_size
		self.kernel_size = params.kernel_size
		self.stride = params.stride
		
		self.embedding = nn.Embedding(self.vocab_size, self.embedding_size, padding_idx=0)
		self.conv1d = nn.Conv1d(self.seq_len, self.out_size, self.kernel_size, self.stride)
		
	def forward(self, x):
	
		print(x.shape)
		out = self.embedding(x)
		print(out.shape)
		out = self.conv1d(out)
		print(out.shape)

		return out