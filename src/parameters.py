from pathlib import Path
from dataclasses import dataclass


@dataclass(init=True, repr=True, eq=True, order=False, unsafe_hash=False, frozen=False)
class Parameters:
    seq_len: int = 35
    num_words: int = 2000

    # Model parameters
    embedding_size: int = 64
    out_size: int = 32
    stride: int = 2

    # Training parameters
    epochs: int = 10
    batch_size: int = 12
    learning_rate: float = 0.001

    data_dir: Path = Path(__file__).parent.parent / 'data'
