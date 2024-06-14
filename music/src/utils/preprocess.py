import numpy as np
from src.constants import NOTE_LIST, OUTPUT_SIZE

def one_hot_encode(key: int) -> np.ndarray:
    v = np.zeros(OUTPUT_SIZE)
    v[NOTE_LIST.index(key)] = 1
    return v

def one_hot_decode(v: np.ndarray) -> int:
    idx = int(np.where(v == 1)[0])
    return NOTE_LIST[idx]