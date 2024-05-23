import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

class DataPreprocessing:
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
        current_symbol = voice[0]
        start_idx = 0
        for idx, next_symbol in enumerate(voice):
            if next_symbol != current_symbol:
                frequency = self.base_freq * 2 ** ((current_symbol - 69) / 12)
                soundvector[start_idx*ticks_per_symbol, idx * ticks_per_symbol]
                
        


start_symbol_index = 1

for n in range(symbolic_length):
    if voice[n] != current_symbol:
        stop_symbol_index = n - 1
        covered_sound_vector_indices = slice((start_symbol_index - 1) * ticks_per_symbol, stop_symbol_index * ticks_per_symbol)
        tone_length = covered_sound_vector_indices.stop - covered_sound_vector_indices.start
        frequency = base_freq * 2 ** ((current_symbol - 69) / 12)
        t = np.arange(tone_length)
        tone_vector = np.sin(2 * np.pi * frequency * t / sample_rate)
        soundvector1[covered_sound_vector_indices] = tone_vector
        current_symbol = voice[n]
        start_symbol_index = n + 1  # Increment to next index

# Handle the last tone
covered_sound_vector_indices = slice((start_symbol_index - 1) * ticks_per_symbol, symbolic_length * ticks_per_symbol)
tone_length = covered_sound_vector_indices.stop - covered_sound_vector_indices.start
frequency = base_freq * 2 ** ((current_symbol - 69) / 12)
t = np.arange(tone_length)
tone_vector = np.sin(2 * np.pi * frequency * t / sample_rate)
soundvector1[covered_sound_vector_indices] = tone_vector

# Normalize sound vector to avoid clipping
soundvector1 = soundvector1 / np.max(np.abs(soundvector1))

# Save to a wav file
write('output.wav', sample_rate, (soundvector1 * 32767).astype(np.int16))