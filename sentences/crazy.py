from sentence_model import SentenceModel
import os

model = SentenceModel()
model.load(os.path.join(os.curdir, "models/"), 'crazy')

string = input("Input next starting sequence. Input '!' to stop the program: \n")
predicted = model.predict_continuous(string)
