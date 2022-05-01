import torch

from torch.utils.data import Dataset


class DatasetMapper(Dataset):

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class Run:
    '''Training, evaluation and metrics calculation'''
    @staticmethod
    def evaluation(model, loader_test):

        # Set the model in evaluation mode
        model.eval()
        predictions = []

        # Starst evaluation phase
        with torch.no_grad():
            for x_batch, y_batch in loader_test:
                y_pred = model(x_batch)
                predictions += list(y_pred.detach().numpy())
        return predictions

    @staticmethod
    def calculate_accuracy(ground_truth, predictions):
        # Metrics calculation
        true_positives = 0
        true_negatives = 0
        for true, pred in zip(ground_truth, predictions):
            if (pred >= 0.5) and (true == 1):
                true_positives += 1
            elif (pred < 0.5) and (true == 0):
                true_negatives += 1
            else:
                pass
        # Return accuracy
        return (true_positives + true_negatives) / len(ground_truth)
