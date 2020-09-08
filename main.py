

from src import Parameters
from src import Preprocessing
from src import TextClassifier
from src import Run
		

class Controller(Parameters):
	
	def __init__(self):
		self.data = self.prepare_data(Parameters.num_words, Parameters.seq_len)
		self.model = TextClassifier(Parameters)
		
		self.training_pipeline(self.model, self.data, Parameters)
		
	@staticmethod
	def training_pipeline(model, data, params):
		run = Run()
		run.train(model, data, params)
		
		
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