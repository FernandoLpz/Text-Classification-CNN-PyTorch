import numpy as np
import nltk
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

from src import Preprocessing
from src import TextClassifier

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from src import Parameters
from src import Data


# class DatasetMaper(Dataset):
# 	'''
# 	Handles batches of dataset
# 	'''
# 	def __init__(self, x, y):
# 		self.x = x
# 		self.y = y
		
# 	def __len__(self):
# 		return len(self.x)
		
# 	def __getitem__(self, idx):
# 		return self.x[idx], self.y[idx]
		

class Controller(Data):
	'''
	Class for execution. Initializes the preprocessing as well as the 
	text classifier model
	'''

	def __init__(self):
		super(Data, self).__init__()
		self.x = Data.x
		self.y = Data.y
		self.vocabulary = Data.vocabulary
		
		print(self.x.shape)
		# self.model = TextClassifier()
		
		
	# def train(self):
		
	# 	training_set = DatasetMaper(self.x_train, self.y_train)
	# 	test_set = DatasetMaper(self.x_test, self.y_test)
		
	# 	self.loader_training = DataLoader(training_set, batch_size=self.batch_size)
	# 	self.loader_test = DataLoader(test_set)
		
	# 	optimizer = optim.RMSprop(self.model.parameters(), lr=args.learning_rate)
	# 	for epoch in range(args.epochs):
			
	# 		predictions = []
			
	# 		self.model.train()
			
	# 		for x_batch, y_batch in self.loader_training:
				
	# 			x = x_batch.type(torch.LongTensor)
	# 			y = y_batch.type(torch.FloatTensor)
				
	# 			y_pred = self.model(x)
				
	# 			loss = F.binary_cross_entropy(y_pred, y)
				
	# 			optimizer.zero_grad()
				
	# 			loss.backward()
				
	# 			optimizer.step()
				
	# 			predictions += list(y_pred.squeeze().detach().numpy())
			
	# 		test_predictions = self.evaluation()
			
	# 		train_accuary = self.calculate_accuray(self.y_train, predictions)
	# 		test_accuracy = self.calculate_accuray(self.y_test, test_predictions)
			
	# 		print("Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f" % (epoch+1, loss.item(), train_accuary, test_accuracy))
			
	# def evaluation(self):

	# 	predictions = []
	# 	self.model.eval()
	# 	with torch.no_grad():
	# 		for x_batch, y_batch in self.loader_test:
	# 			x = x_batch.type(torch.LongTensor)
	# 			y = y_batch.type(torch.FloatTensor)
				
	# 			y_pred = self.model(x)
	# 			predictions += list(y_pred.detach().numpy())
				
	# 	return predictions
			
	# @staticmethod
	# def calculate_accuray(grand_truth, predictions):
	# 	true_positives = 0
	# 	true_negatives = 0
		
	# 	for true, pred in zip(grand_truth, predictions):
	# 		if (pred > 0.5) and (true == 1):
	# 			true_positives += 1
	# 		elif (pred < 0.5) and (true == 0):
	# 			true_negatives += 1
	# 		else:
	# 			pass
				
	# 	return (true_positives+true_negatives) / len(grand_truth)
	
if __name__ == "__main__":
	
	control = Controller()