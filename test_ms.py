import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import torchaudio
import numpy as np


fn = "/home/evelyn/Documents/Job Applications/SpeechGraphics/SG_tech_test/ML_exercice/data/singing/sing030/sing030_10.wav"

# wav, sr = librosa.load(fn, sr=16000)
# wav = librosa.effects.preemphasis(wav) # pre-emphasis is messing up singing spectrograms - huge blown-out high frequencies
wav, sr = torchaudio.load(fn, normalize=True)
transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=2048, hop_length=265, n_mels=128)
ms = transform(wav)
log_ms = librosa.amplitude_to_db(ms.squeeze(0))
mm_scaler = MinMaxScaler()
mm_scaler.fit(log_ms)#.squeeze(0))
scaled_log_ms = mm_scaler.transform(log_ms)#.squeeze(0))
scaled_log_ms = torch.Tensor(scaled_log_ms)
print(type(scaled_log_ms))
plt.imshow(scaled_log_ms, aspect='auto')
plt.gca().invert_yaxis()
plt.show()

