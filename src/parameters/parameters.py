from dataclasses import dataclass

@dataclass
class Parameters:
   seq_len: int = 35
   num_words: int = 2000
   batch_size: int = 128
   embedding_size: int = 64
   out_size: int = 32
   kernel_size: int = 3
   stride: int = 2