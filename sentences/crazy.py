from sentence_model import SentenceModel
import os

model = SentenceModel(hidden_size=256)
dir_path = os.path.dirname(os.path.realpath(__file__))
model.load(os.path.join(dir_path, "models/"), 'crazySGD')

string = input("Input next starting sequence. Input '!' to stop the program: \n")
predicted = model.predict_continuous(string)
