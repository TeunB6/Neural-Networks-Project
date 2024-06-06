import os
from src.voice_loader import VoiceLoader
from sklearn.model_selection import train_test_split

def train():
    path = os.path.dirname(os.path.realpath(__file__))

    
    parameters = {"base_freq" : 440,
                  "sample_rate" : 10000,
                  "duration_per_symbol" : 1/16}
    vl = VoiceLoader(parameters)
    data = vl(path)
    
    # Create subsequences and next note
    X = None
    y = None
    
    X_train, X_test, y_train, y_test = train_test_split(data)
    