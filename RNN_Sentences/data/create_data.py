import sys
import os
dir_path = os.path.dirname(os.path.realpath(__file__))


def create_dataset(sentence):
    data = []
    length = len(sentence)
    for shift in range(length):
        end = sentence[:shift]
        start = sentence[shift:]
        data.append(start + end)
    return data
def main(name, sentence):
    data = create_dataset(sentence)
    with open(os.path.join(dir_path, f"{name}.txt"), 'a') as file:
        for line in data:
            file.write(line + '\n')

if __name__ == "__main__":
    args = sys.argv[1:]
    if len(args) != 2:
        ValueError("Incorrect usage: python create_data.py name sentence")

    name, sentence = args
    main(name, sentence)