import torch
import torch.optim as optim
import torch.nn.functional as F

from src import Parameters
from src import Preprocessing
from src import TextClassifier

from torch.utils.data import Dataset, DataLoader

class DatasetMaper(Dataset):

	def __init__(self, x, y):
		self.x = x
		self.y = y
		
	def __len__(self):
		return len(self.x)
		
	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]
		

class Controller(Parameters):
	
	def __init__(self):
		self.data = self.prepare_data(Parameters.num_words, Parameters.seq_len)
		self.model = TextClassifier(Parameters)
		self.train(self.model, self.data, Parameters)
		
	@staticmethod
	def train(model, data, params):
	
		train = DatasetMaper(data['x_train'], data['y_train'])
		test = DatasetMaper(data['x_test'], data['y_test'])
		
		loader_train = DataLoader(train, batch_size=params.batch_size)
		loader_test = DataLoader(test, batch_size=params.batch_size)
		
		optimizer = optim.RMSprop(model.parameters(), lr=params.learning_rate)
		
		for epoch in range(params.epochs):
		
			model.train()
			predictions = []
			
			for x_batch, y_batch in loader_train:
				
				y_batch = y_batch.type(torch.FloatTensor)
				y_pred = model(x_batch)
				loss = F.binary_cross_entropy(y_pred, y_batch)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
				predictions += list(y_pred.detach().numpy())
				
			test_predictions = Controller.evaluation(model, loader_test)
			
			train_accuary = Controller.calculate_accuray(data['y_train'], predictions)
			test_accuracy = Controller.calculate_accuray(data['y_test'], test_predictions)
			
			print("Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f" % (epoch+1, loss.item(), train_accuary, test_accuracy))
		
	@staticmethod
	def evaluation(model, loader_test):
		model.eval()
		predictions = []
		
		with torch.no_grad():
			for x_batch, y_batch in loader_test:
				y_pred = model(x_batch)
				predictions += list(y_pred.detach().numpy())
				
		return predictions
		
	@staticmethod
	def calculate_accuray(grand_truth, predictions):
		true_positives = 0
		true_negatives = 0
		
		for true, pred in zip(grand_truth, predictions):
			if (pred >= 0.5) and (true == 1):
				true_positives += 1
			elif (pred < 0.5) and (true == 0):
				true_negatives += 1
			else:
				pass
				
		return (true_positives+true_negatives) / len(grand_truth)
		
	@staticmethod
	def prepare_data(num_words, seq_len):
		pr = Preprocessing(num_words, seq_len)
		pr.load_data()
		pr.clean_text()
		pr.text_tokenization()
		pr.build_vocabulary()
		pr.word_to_idx()
		pr.padding_sentences()
		pr.split_data()

		return {'x_train': pr.x_train, 'y_train': pr.y_train, 'x_test': pr.x_test, 'y_test': pr.y_test}
		
if __name__ == '__main__':
	controller = Controller()