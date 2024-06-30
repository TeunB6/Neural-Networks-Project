import numpy as np
from src.constants import NOTE_LIST, OUTPUT_SIZE

def one_hot_encode(key: int) -> np.ndarray:
    """
    Create a one hot encoding based on the provided key/note

    Args:
        key (int): the key to encode

    Returns:
        np.ndarray: the resulting one hot encoding
    """    
    v = np.zeros(OUTPUT_SIZE)
    v[NOTE_LIST.index(key)] = 1
    return v

def one_hot_decode(v: np.ndarray) -> int:
    """
    Decodes a one hot encoded array using the NOTE_LIST constant containing the index of every note in the one hot encoded vector

    Args:
        v (np.ndarray): the one hot encoded vector to decode

    Returns:
        int: the note encoded
    """    
    idx = int(np.where(v == 1)[0])
    return NOTE_LIST[idx]

def one_hot_max(arr) -> np.ndarray:
    """
    Turns the provided np.ndarray into a one hot encoding of that array where the first max is marked with a 1

    Args:
        arr (np.ndarray): the array to one hot encode using maxout

    Returns:
        np.ndarray: the resulting one hot encoding
    """  
    max_idx = np.argmax(arr)
    one_hot = np.zeros_like(arr)
    one_hot[max_idx] = 1
    return one_hot
