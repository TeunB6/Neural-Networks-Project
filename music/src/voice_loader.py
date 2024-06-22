import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from typing import Optional
import os

class VoiceLoader:
    def __init__(self, parameters: dict = {}) -> None:
        # base parameters
        self.base_freq = parameters.get("base_freq", 440)
        self.sample_rate = parameters.get("sample_rate", 10000)
        self.duration_per_symbol = parameters.get("duration_per_symbol", 1/16)
        
    def _load_data(self, path: str):
        data = []
        with open(path, 'r') as f:
            for line in f.readlines():
                data.append(list(map(int, line.split())))
        return np.asarray(data)

    def _load_voice(self, data: np.ndarray, voice: int):
        voice_data = data[:, voice] if data.ndim == 2 else data
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
            
            fade_in_out_length = min(int(0.01 * self.sample_rate), tone_length // 2)
            fade_in = np.linspace(0, 1, fade_in_out_length)
            fade_out = np.linspace(1, 0, fade_in_out_length)
            tone_vector[:fade_in_out_length] *= fade_in
            tone_vector[-fade_in_out_length:] *= fade_out
            
            
            soundvector[start_idx*ticks_per_symbol:idx * ticks_per_symbol] = tone_vector
            current_symbol = next_symbol
            start_idx = idx
        return soundvector
    
    def __call__(self, save_path: str, data_path: Optional[str] = None, voice: Optional[int] = None, data: Optional[np.ndarray] = None, name: str = ""): # TODO: make this work with providing the voice directly
        if data is None:
            voice_data = self._load_data(data_path)
        else:
            voice_data = data
        
        soundvector = self._load_voice(voice_data, voice)
        
        
        # if not os.path.exists(output_path):
        output_path = os.path.join(save_path, f"output_{name}.wav")
        write(output_path, self.sample_rate, (soundvector * 32767).astype(np.int16))

if __name__ == "__main__":
    path = '/home/teunb/Assignments/Yr2 - Assignments/Neural-Networks-Project/music/ProjectBach_materials/F.txt'
    parameters = {"base_freq" : 440,
                  "sample_rate" : 10000,
                  "duration_per_symbol" : 1/16}

    vl = VoiceLoader(parameters)
    vl(path, 0)
