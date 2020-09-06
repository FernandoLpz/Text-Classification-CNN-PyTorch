import torch
import torch.nn as nn
import torch.nn.functional as F

class TextClassifier(nn.ModuleList):

	def __init__(self, params):
		super(TextClassifier, self).__init__()
		
		self.batch_size = params.batch_size
		self.seq_len = params.seq_len
		self.num_words = params.num_words
		self.embedding_size = params.embedding_size
		
		self.out_size = params.out_size
		self.kernel_1 = 2
		self.kernel_2 = 3
		self.kernel_3 = 4
		
		self.kernel_size = params.kernel_size
		self.stride = params.stride
		
		self.embedding = nn.Embedding(self.num_words, self.embedding_size, padding_idx=0)
		self.conv_1 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_1, self.stride)
		self.conv_2 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_2, self.stride)
		self.conv_3 = nn.Conv1d(self.seq_len, self.out_size, self.kernel_3, self.stride)
		self.pool_1 = nn.MaxPool1d(self.kernel_1, self.stride)
		self.pool_2 = nn.MaxPool1d(self.kernel_2, self.stride)
		self.pool_3 = nn.MaxPool1d(self.kernel_3, self.stride)
		
		
	def forward(self, x):
	
		print(f"x raw: {x.shape}")
		x = self.embedding(x)
		print(f"x embedded: {x.shape}")
		
		x1 = self.conv_1(x)
		print(f"\nx1 convolved: {x1.shape}")
		x1 = self.pool_1(x1)
		print(f"x1 pooled: {x1.shape}")
		
		x2 = self.conv_2(x)
		print(f"\nx2 convolved: {x2.shape}")
		x2 = self.pool_2(x2)
		print(f"x2 pooled: {x2.shape}")
		
		x3 = self.conv_3(x)
		print(f"\nx3 convolved: {x3.shape}")
		x3 = self.pool_3(x3)
		print(f"x3 pooled: {x3.shape}")
		

		return 1