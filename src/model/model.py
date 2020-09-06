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
		
		self.fc = nn.Linear(1440, 1)
		
		
	def forward(self, x):
	
		x = self.embedding(x)
		
		x1 = self.conv_1(x)
		x1 = torch.relu(x1)
		x1 = self.pool_1(x1)
		
		x2 = self.conv_2(x)
		x2 = torch.relu((x2))
		x2 = self.pool_2(x2)
	
		x3 = self.conv_3(x)
		x3 = torch.relu(x3)
		x3 = self.pool_3(x3)
		
		union = torch.cat((x1, x2, x3), 2)
		union = union.reshape(union.size(0), -1)
		
		out = self.fc(union)
		out = torch.sigmoid(out)
		
		return out
