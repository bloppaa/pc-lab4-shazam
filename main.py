import matplotlib.pyplot as plt
import librosa
from skimage import img_as_float
from spectrogram import generate_spectrogram
from fingerprint import find_local_peaks, generate_hashes

spectrogram, sr = generate_spectrogram()
peaks = find_local_peaks(spectrogram)
hashes = generate_hashes(peaks)

print(spectrogram)
print(sr)
print(peaks)
print(hashes)
