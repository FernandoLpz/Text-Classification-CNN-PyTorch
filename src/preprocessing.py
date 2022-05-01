import re
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.parameters import Parameters


def word_tokenize(text):
    return re.findall(r'\w+', text)


class LabeledTexts:

    def __init__(self, data_filepath='tweets.csv', parameters=Parameters()):
        self.num_words = parameters.num_words
        self.seq_len = parameters.seq_len
        self.data_dir = parameters.data_dir
        self.parameters = parameters

        self.data_filepath = Path(data_filepath)
        if not self.data_filepath.is_file():
            self.data_filepath = self.data_dir / (data_filepath or 'tweets.csv')

        self.vocabulary = None
        self.x_idnums = None
        self.x_padded = None
        self.x = None
        self.y = None

        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        # Reads the raw csv file and split into texts (sentences) (x) and target label (y)
        df = pd.read_csv(self.data_filepath.open())
        df.drop(['id', 'keyword', 'location'], axis=1, inplace=True)

        self.x = df['text'].values
        self.y = df['target'].values

    def clean_text(self):
        # Removes special symbols and just keep
        # words in lower or upper form

        self.x = [x.lower() for x in self.x]
        self.x = [re.sub(r'[^A-Za-z]+', ' ', x) for x in self.x]

    def text_tokenization(self):
        # Tokenizes each sentence by implementing the nltk tool
        self.x = [word_tokenize(x) for x in self.x]

    def build_vocabulary(self):
        # Builds the vocabulary and keeps the "x" most frequent words
        self.vocabulary = dict()
        fdist = Counter()

        for sentence in self.x:
            for word in sentence:
                fdist[word] += 1

        common_words = fdist.most_common(self.num_words)

        for idx, word in enumerate(common_words):
            self.vocabulary[word[0]] = (idx + 1)

    def word_to_idx(self):
        # By using the dictionary (vocabulary), it is transformed
        # each token into its index based representation

        self.x_idnums = list()

        for sentence in self.x:
            temp_sentence = list()
            for word in sentence:
                if word in self.vocabulary.keys():
                    temp_sentence.append(self.vocabulary[word])
            self.x_idnums.append(temp_sentence)

    def padding_sentences(self):
        # Each sentence which does not fulfill the required len
        # it's padded with the index 0

        pad_idx = 0
        self.x_padded = list()

        for sentence in self.x_idnums:
            while len(sentence) < self.seq_len:
                sentence.insert(len(sentence), pad_idx)
            self.x_padded.append(sentence)

        self.x_padded = np.array(self.x_padded)

    def split_data(self):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x_padded, self.y, test_size=0.25, random_state=42)
