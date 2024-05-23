import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
import os

class VoiceLoader:
    def __init__(self, parameters: dict) -> None:
        # base parameters
        self.base_freq = parameters["base_freq"] if parameters["base_freq"] else 440
        self.sample_rate = parameters["sample_rate"] if parameters["sample_rate"] else 10000
        self.duration_per_symbol = parameters["duration_per_symbol"] if parameters["duration_per_symbol"] else 1/16

    def _load_data(self, path: str):
        data = []
        with open(path, 'r') as f:
            for line in f.readlines():
                data.append(line.split())
        return np.asarray(data)

    def _load_voice(self, data: np.ndarray, voice: int):
        voice_data = data[:, voice]
        symbolic_length = len(voice_data)
        ticks_per_symbol = int(self.sample_rate * self.duration_per_symbol)
        
        # initialize loop
        soundvector = np.zeros(symbolic_length * ticks_per_symbol)
        current_symbol = voice_data[0]
        start_idx = 0
        for idx, next_symbol in enumerate(voice_data):
            if next_symbol == current_symbol or idx == symbolic_length - 1:
                continue
            
            frequency = self.base_freq * 2 ** ((current_symbol - 69) / 12)
            tone_length = idx * ticks_per_symbol - start_idx * ticks_per_symbol
    
            t = np.arange(tone_length)
            
            tone_vector = np.sin(2*np.pi*frequency*t / self.sample_rate)
            soundvector[start_idx*ticks_per_symbol, idx * ticks_per_symbol] = tone_vector
            current_symbol = next_symbol
            start_idx = idx
        return soundvector
    
    def __call__(self, path: str, voice: int) -> np.Any:
        dir = os.path.splitdrive(path)[0]
        output_path = os.path.join(dir, f"output_voice_{voice}.wav")
        
        voice_data = self._load_data(path)
        soundvector = self._load_voice(path, voice)
        
        
        
        if not os.path.exists(output_path):
            write(output_path, self.sample_rate, (soundvector * 32767).astype(np.int16))
        
