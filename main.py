from src import Parameters
from src import Preprocessing
from src import TextClassifier
from src import Run


class Controller(Parameters):

    def __init__(self):
        # Preprocessing pipeline
        self.data = self.prepare_data(Parameters.num_words, Parameters.seq_len)
        self.x_train = self.data['x_train']
        self.y_train = self.data['y_train']
        self.x_test = self.data['x_test']
        self.y_test = self.data['y_test']

        # Initialize the model
        self.model = TextClassifier(Parameters)

        # Training - Evaluation pipeline

    def prepare_data(self):
        # Preprocessing pipeline
        pr = Preprocessing(self.num_words, self.seq_len)
        pr.load_data()
        pr.clean_text()
        pr.text_tokenization()
        pr.build_vocabulary()
        pr.word_to_idx()
        pr.padding_sentences()
        pr.split_data()

        return {'x_train': pr.x_train, 'y_train': pr.y_train, 'x_test': pr.x_test, 'y_test': pr.y_test}

    def train(self):
        # Initialize dataset maper
        train = DatasetMapper(self.x_train, self.y_train)
        test = DatasetMapper(data['x_test'], data['y_test'])

        # Initialize loaders
        loader_train = DataLoader(train, batch_size=params.batch_size)
        loader_test = DataLoader(test, batch_size=params.batch_size)

        # Define optimizer
        optimizer = optim.RMSprop(model.parameters(), lr=params.learning_rate)

        # Starts training phase
        for epoch in range(params.epochs):
            # Set model in training model
            model.train()
            predictions = []
            # Starts batch training
            for x_batch, y_batch in loader_train:

                y_batch = y_batch.type(torch.FloatTensor)

                # Feed the model
                y_pred = model(x_batch)

                # Loss calculation
                loss = F.binary_cross_entropy(y_pred, y_batch)

                # Clean gradientes
                optimizer.zero_grad()

                # Gradients calculation
                loss.backward()

                # Gradients update
                optimizer.step()

                # Save predictions
                predictions += list(y_pred.detach().numpy())

            # Evaluation phase
            test_predictions = Run.evaluation(model, loader_test)

            # Metrics calculation
            train_accuary = Run.calculate_accuray(data['y_train'], predictions)
            test_accuracy = Run.calculate_accuray(data['y_test'], test_predictions)
            print("Epoch: %d, loss: %.5f, Train accuracy: %.5f, Test accuracy: %.5f" % (epoch + 1, loss.item(), train_accuary, test_accuracy))

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

    def predict(self, text):
        return self.model.predict(text)


if __name__ == '__main__':
    controller = Controller()
