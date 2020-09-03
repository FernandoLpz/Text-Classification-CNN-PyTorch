from preprocessor import Preprocessing 
from dataclasses import dataclass

@dataclass
class Parameters:
   seq_len: int = 35
   num_words: int = 2000
   batch_size: int = 64
   learning_rate: int = 0.01

class Data:
   def __init__(self):
      self.x = None
      self.y = None
      self.vocabulary = None
      
      self.preprocess_data()
      
   def preprocess_data(self):
      prep = Preprocessing()
      prep.load_data()
      prep.clean_text()
      prep.text_tokenization()
      prep.build_vocabulary()
      prep.word_to_idx()
      prep.padding_sentences()
      
      self.x = prep.x_padded
      self.y = prep.y
      self.vocabulary = prep.vocabulary
      
if __name__ == '__main__':
   data = Data()
   x = data.x
   y = data.y