import numpy as np
from src.constants import NOTE_LIST, INPUT_SIZE

def one_hot_encode(key: int) -> np.ndarray:
    v = np.zeros(len(NOTE_LIST))
    v[NOTE_LIST.index(key)] = 1
    return v

def one_hot_decode(v: np.ndarray) -> int:
    idx = np.where(v == 1)
    return NOTE_LIST[INPUT_SIZE]