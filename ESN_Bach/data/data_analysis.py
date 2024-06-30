import matplotlib.pyplot as plt
import os

path = os.path.dirname(os.path.realpath(__file__))

def load_column(c, file):
    with open(os.path.join(path, file)) as f:
        return [int(line.split()[c]) for line in f.readlines()]

def histogram(data: list, axis_type: type = int) -> list:
    unique = set(data)
    return [axis_type(u) for u in unique], [data.count(item) for item in unique]

voice_data = load_column(0, 'F.txt')
processed_data = load_column(0, 'sequence.txt')

voice_hist = histogram(voice_data, str)
processed_hist = histogram(processed_data, str)

plt.figure()
plt.bar(*voice_hist, label="Original key count")
plt.bar(*processed_hist, label="Preprocessed key count")
plt.legend()
plt.title("Histogram of the original and preprocessed keys")
plt.show()

durr_data = load_column(1, 'sequence.txt')
durr_hist = histogram(durr_data)

plt.figure()
plt.bar(*durr_hist, width=2)
plt.title("Histogram of the duration in the preprocessed data")

plt.show()