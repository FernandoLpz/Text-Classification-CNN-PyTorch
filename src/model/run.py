import torch
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

class DatasetMaper(Dataset):

	def __init__(self, x, y):
		self.x = x
		self.y = y
		
	def __len__(self):
		return len(self.x)
		
	def __getitem__(self, idx):
		return self.x[idx], self.y[idx]

class Run:

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
         test_predictions = Run.evaluation(model, loader_test)
         train_accuary = Run.calculate_accuray(data['y_train'], predictions)
         test_accuracy = Run.calculate_accuray(data['y_test'], test_predictions)
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
		