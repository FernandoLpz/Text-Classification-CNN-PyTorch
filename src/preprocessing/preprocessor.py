import re
import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
# from src.parameters import Parameters

class Preprocessing:
	
	def __init__(self):
		self.data = '../data/tweets.csv'
		self.vocabulary = None
		self.x_tokenized = None
		self.x_padded = None
		self.x_raw = None
		self.y = None
		
# 	def load_data(self):
# 		df = pd.read_csv(self.data)
# 		df.drop(['id','keyword','location'], axis=1, inplace=True)
		
# 		self.x_raw = df['text'].values
# 		self.y = df['target'].values
		
# 	def clean_text(self):
# 		self.x_raw = [x.lower() for x in self.x_raw]
# 		self.x_raw = [re.sub(r'[^A-Za-z]+', ' ', x) for x in self.x_raw]
		
# 	def text_tokenization(self):
# 	   self.x_raw = [word_tokenize(x) for x in self.x_raw]
	   
# 	def build_vocabulary(self):
# 	   self.vocabulary = dict()
# 	   fdist = nltk.FreqDist()
	   
# 	   for sentence in self.x_raw:
# 	      for word in sentence:
# 	         fdist[word] += 1
	         
# 	   common_words = fdist.most_common(Parameters.num_words)
	   
# 	   for idx, word in enumerate(common_words):
# 	      self.vocabulary[word[0]] = idx
	      
# 	def word_to_idx(self):
# 	   self.x_tokenized = list()
	   
# 	   for sentence in self.x_raw:
# 	      temp_sentence = list()
# 	      for word in sentence:
# 	         if word in self.vocabulary.keys():
# 	            temp_sentence.append(self.vocabulary[word])
# 	      self.x_tokenized.append(temp_sentence)
	      
# 	def padding_sentences(self):
# 	   pad_idx = Parameters.num_words + 1
# 	   self.x_padded = list()
	   
# 	   for sentence in self.x_tokenized:
# 	      while len(sentence) < Parameters.seq_len:
# 	         sentence.insert(len(sentence), pad_idx)
# 	      self.x_padded.append(sentence)
	   
# 	   self.x_padded = np.array(self.x_padded)
	   
# class Data:
#    def __init__(self):
#       self.x = None
#       self.y = None
#       self.vocabulary = None
      
#       self.preprocess_data()
      
#    def preprocess_data(self):
   
#       prep = Preprocessing()
#       prep.load_data()
#       prep.clean_text()
#       prep.text_tokenization()
#       prep.build_vocabulary()
#       prep.word_to_idx()
#       prep.padding_sentences()
      
#       self.x = prep.x_padded
#       self.y = prep.y
#       self.vocabulary = prep.vocabulary