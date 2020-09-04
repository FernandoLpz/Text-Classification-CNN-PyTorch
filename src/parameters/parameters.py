from dataclasses import dataclass

@dataclass
class Parameters:
   seq_len: int = 35
   num_words: int = 2000
   batch_size: int = 64
   learning_rate: int = 0.01