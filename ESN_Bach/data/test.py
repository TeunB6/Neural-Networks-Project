import os

path = os.path.dirname(os.path.realpath(__file__))

sentence = ""
alphabet = "abcdefghijklmnopqrstuvwxyz"

NOTE_LIST = [0, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 73, 74, 76]


with open(os.path.join(path, 'sequence.txt'), 'r') as f:
    sequence = [tuple(map(int, line.split())) for line in f.readlines()]
    for note, _ in sequence:
        sentence += alphabet[NOTE_LIST.index(note)]

with open(os.path.join(path, 'sentence_test.txt'), 'w') as f:
    f.write(sentence)