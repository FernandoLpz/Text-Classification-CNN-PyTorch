from dataclasses import dataclass

@dataclass
class Parameters:
   seq_len: int = 35
   num_words: int = 2000
   epochs: int = 10
   batch_size: int = 12
   learning_rate: float = 0.001
   embedding_size: int = 128
   out_size: int = 32
   kernel_size: int = 3
   stride: int = 2