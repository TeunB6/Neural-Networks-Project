from sentence_model import SentenceModel
from itertools import combinations
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

def train_new(name):
    path = os.path.join(os.curdir, "data")

    raw_sentences = []

    file = "crazy.txt" # fill in data file you want to train on here
    
    fp = os.path.join(path, file)
    with open(fp, 'r') as f:
        raw_sentences += [line.strip() for line in f.readlines()]

    def extract_data(sentences: list[str]) -> tuple[list[str], list[str]]:
        strings, chars = [], []
        for s in sentences:
            sub_strings = [s[i:j] for i, j in combinations(range(len(s) + 1), r=2) if len(s[i:j]) >= 2]
            for sub in sub_strings:
                strings.append(sub[:-1])
                chars.append(sub[-1])
        return strings, chars

    X, y = extract_data(raw_sentences)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = SentenceModel()
    model.train(X_train, y_train)

    y_pred = [model.predict_char(s) for s in X_test]

    model.save(os.path.join(os.curdir, "models/"), name)

    print("Accuracy: ", accuracy_score(y_test, y_pred))

def run_existing(name):
    model = SentenceModel()
    model.load(os.path.join(os.curdir, "models/"), name)

    while True:
        string = input("Input next starting sentence. Input '!' to stop the program: \n")
        if string == '!':
            exit(0)
        
        predicted = model.predict_sentence(string)
        print(f"Predicted sentence: {predicted}")

if __name__ == "__main__":
    import sys
    args = sys.argv
    if len(args) < 2 or len(args) > 3:
        raise ValueError("Invalid command line run: python main.py train | run name")    
    if args[1] == 'train':
        train_new(args[2])
    elif args[1] == 'run':
        run_existing(args[2])
    else:
        raise ValueError("Invalid command line run: python main.py train | run name")