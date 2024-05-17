from sentence_model import SentenceModel
import os

model = SentenceModel(hidden_size=256)
model.load(os.path.join(os.curdir, "models/"), 'crazy2')

string = input("Input next starting sequence. Input '!' to stop the program: \n")
predicted = model.predict_continuous(string)
