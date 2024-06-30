import numpy as np
import os
from src.constants import NOTE_LIST

path = "/home/teunb/Assignments/Yr2 - Assignments/Neural-Networks-Project/music/data/F.txt"

def extract_data():
    size_min = 2 # min is 2 : large, small and crazy were all trained with size_threshold = 2
    size_max = 20
    data = []
    with open(path, 'r') as f:
        for line in f.readlines():
            data.append(list(map(int, line.split())))
    
    dir = os.path.dirname(path)     
    data = np.asarray(data)[:, 0]
    return data

labels = extract_data()
n_samples = len(labels)
n_classes = 22
hist = {}
weight = []
for key in NOTE_LIST:
    hist[key] = np.count_nonzero(labels == key)
    weight.append(n_samples / (np.count_nonzero(labels == key) * n_classes))
print(hist)
print("WEIGHT_VECTOR = ", weight)
    