import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from typing import Optional
import os

class VoiceLoader:
    """Class implementing functionality for loading the voice from the disk or saving time series on the disk as .wav files
    """    
    def __init__(self, base_freq: int = 440, sample_rate: int = 10000, duration_per_symbol: float = 1/16 ) -> None:
        # Initialize parameters needed to create the sound file
        self.base_freq = base_freq
        self.sample_rate = sample_rate
        self.duration_per_symbol = duration_per_symbol
        
    def _load_data(self, path: str) -> np.ndarray:
        """
        Load data stored at path from the disk

        Args:
            path (str): path storing the time series data

        Returns:
            np.ndarray: the loaded timeseries
        """        
        data = []
        with open(path, 'r') as f:
            for line in f.readlines():
                data.append(list(map(int, line.split())))
        return np.asarray(data)

    def _load_voice(self, data: np.ndarray, voice: Optional[int] = None) -> np.ndarray:
        """Loads the provided voice from the data creating a .wav file which can be listened to

        Args:
            data (np.ndarray): the data containing the time series
            voice (int, optional): the voice to filter from the provided data, if data contains 1 voice this is unused. Defaults to None

        Returns:
            np.ndarray: the resulting timeseries in the sound domain
        """        
        voice_data = data[:, voice] if data.ndim == 2 else data
        symbolic_length = len(voice_data)
        ticks_per_symbol = int(self.sample_rate * self.duration_per_symbol)
        
        # Initialize loop
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
    
    def __call__(self, save_path: str, data_path: Optional[str] = None,
                 voice: Optional[int] = None, data: Optional[np.ndarray] = None, name: str = "") -> None: 
        """Runs the voice loader loading data from the provided path or np.ndarray and saving the .wav file at save_path under the provided name

        Args:
            save_path (str): The location to save the file at
            data_path (Optional[str], optional): The location to load the data from (if necessary). Defaults to None.
            voice (Optional[int], optional): The voice to select from the data (if necessary). Defaults to None.
            data (Optional[np.ndarray], optional): The data to save as a .wav file. Defaults to None.
            name (str, optional): The name to save the files under. Defaults to "".
        """        
        if data is None:
            voice_data = self._load_data(data_path)
        else:
            voice_data = data
        
        soundvector = self._load_voice(voice_data, voice)    
        output_path = os.path.join(save_path, f"output_{name}.wav")
        write(output_path, self.sample_rate, (soundvector * 32767).astype(np.int16))

if __name__ == "__main__":
    # Loads the F.txt file
    path = '/home/teunb/Assignments/Yr2 - Assignments/Neural-Networks-Project/music/data/F.txt'
    parameters = {"base_freq" : 440,
                  "sample_rate" : 10000,
                  "duration_per_symbol" : 1/16}

    vl = VoiceLoader(parameters)
    vl(path, 0)
