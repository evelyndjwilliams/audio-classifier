import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import torchaudio
import numpy as np
import argparse
import shutil



# def gen_model(input_dims):


class LSTM(nn.Module):
    """ Defines architecture and training procedure for model.
        Model is a 1-layer LSTM with 128 hidden units."""

    def __init__(self):
        """Model architecture"""
        self.model = self.generate_model(input_size)
        self.forward = self.model.forward()

    def generate_model(input_size):
        return nn.Sequential(
            nn.LSTM(input_size=input_size, hidden_size=128, num_layers=1, batch_first=True), # make sure batch is first dimension in input tensor, or change this
            nn.Linear(128,128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128,3),
            nn.Softmax()
            )


class Preprocessor():

    def __init__(self):
        # self.__init__()
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=2048, hop_length=265, n_mels=128)

    def preprocess(self, filename):
        # wav, sr = librosa.load(fn, sr=16000)
        # wav = librosa.effects.preemphasis(wav) # pre-emphasis is messing up singing spectrograms - huge blown-out high frequencies
        wav, sr = torchaudio.load(filename, normalize=True)
        # transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=2048, hop_length=265, n_mels=128)
        ms = self.transform(wav)
        log_ms = librosa.amplitude_to_db(ms.squeeze(0))
        mm_scaler = MinMaxScaler()
        mm_scaler.fit(log_ms)#.squeeze(0))
        scaled_log_ms = mm_scaler.transform(log_ms)#.squeeze(0))
        # plt.imshow(scaled_log_ms, aspect='auto')
        # plt.gca().invert_yaxis()
        # plt.show()
        return torch.Tensor(scaled_log_ms)

    def preprocess_dir(self, in_dir, out_dir):
        if  os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.mkdir(out_dir)
        for genre in os.listdir(in_dir):
            if genre.startswith('.'):
                continue
            genre_dir = os.path.join(in_dir, genre)
            # if not os.path.exists(os.path.join(out_dir, genre)):
            os.mkdir(os.path.join(out_dir, genre))
            for person in os.listdir(genre_dir):
                person_dir = os.path.join(genre_dir, person)
                for file in os.listdir(person_dir):
                    if file.startswith('.'):
                        continue
                    if file[-4:] == '.wav':
                        file_path = os.path.join(person_dir, file)
                        log_melspec = self.preprocess(file_path)
                        out_name = os.path.join(out_dir, genre, f"{file[:-4]}.pt")
                        print(out_name)
                        torch.save(log_melspec, out_name)

if __name__ == "__main__":
    # args: store_true, if preprocess = True: run preprocesing, else just run training
    parser = argparse.ArgumentParser(description='Audio classifier')
    parser.add_argument('--preprocess', action='store_true')
    args = parser.parse_args()

    if args.preprocess == True:
        data_preprocessor = Preprocessor()
        data_preprocessor.preprocess_dir("ML_exercice/data", "ML_exercice/preprocessed_data/")
    # preprocess_dir(in_dir = )
    # SR = 16000
    # m
# preprocessing data

