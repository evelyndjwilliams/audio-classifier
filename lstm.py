import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import librosa
import librosa.display
import matplotlib.pyplot as plt
import os
import torchaudio
import numpy as np
import argparse
import shutil
import pandas as pd
from torchsummary import summary

class LSTM(nn.Module):
    """ Defines architecture and training procedure for model.
        Model is a 1-layer LSTM with 182 hidden units."""

    def __init__(self):
        """Model architecture"""
        super().__init__()
        # self.forward = self.model.forward()
        self.lstm = nn.LSTM(input_size=80, hidden_size=182, num_layers=1, batch_first=True)
        # make sure batch is first dimension in input tensor, or change this
        self.linear1 = nn.Linear(182,182)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.linear2 = nn.Linear(182,3)
        self.softmax = nn.Softmax(dim=1) # not using softmax because CrossEntropyLoss applies log softmax


    def forward(self, input_data):
        x = self.lstm(input_data)
        print(f"after lstm part 0: {x[0].size()}")
        print(f"after lstm part 1 0 : {x[1][0].size()}")
        print(f"after lstm part 1 1 : {x[1][1].size()}")
        # this is using final hidden state (not cell state) as input to linear
        # as here https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html?highlight=lstm#torch.nn.LSTM
        x = self.linear1(x[1][0])
        print(f"after linear 1: {x[0].size()}")
        x = self.relu(x)
        print(f"after relu: {x[0].size()}")
        x = self.dropout(x)
        print(f"after dropout: {x[0].size()}")
        x = self.linear2(x)
        print(f"after linear 2: {x[0].size()}")
        predictions = x
        return predictions


class Preprocessor():
    """ Extracts scaled log mel-spectrograms from audio dataset.
        Stores in /preprocessed-data 
        To run preprocessing, pass --preprocess arg to train.py"""

    def __init__(self):
        # self.__init__()
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=2048, hop_length=265, n_mels=80)
        self.labels = {'speech':0, 'rap':1, 'singing':2}

    def preprocess(self, filename):
        # wav, sr = librosa.load(fn, sr=16000)
        # wav = librosa.effects.preemphasis(wav) # pre-emphasis is messing up singing spectrograms - huge blown-out high frequencies
        wav, sr = torchaudio.load(filename, normalize=True)
        # transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=2048, hop_length=265, n_mels=182)
        ms = self.transform(wav)
        log_ms = librosa.amplitude_to_db(ms.squeeze(0))
        mm_scaler = MinMaxScaler()
        mm_scaler.fit(log_ms)#.squeeze(0))
        scaled_log_ms = mm_scaler.transform(log_ms)#.squeeze(0))
        # plt.imshow(scaled_log_ms, aspect='auto')
        # plt.gca().invert_yaxis()
        # plt.show()
        return torch.Tensor(scaled_log_ms)

    def preprocess_dir(self, in_dir, out_dir, info_file):
        if  os.path.exists(out_dir):
            shutil.rmtree(out_dir)
        os.mkdir(out_dir)
        with open(info_file, 'w+') as f:
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
                            torch.save(log_melspec.transpose(0,1), out_name)
                            f.write(f"{out_name},{genre},{self.labels[genre]}\n")

class AudioDataset(Dataset):

    def __init__(self, label_file, data_dir):
        self.data_info = pd.read_csv(label_file)
        self.data_dir = data_dir

    def __len__(self):
        return len(self.data_info)

    def __getitem__(self, index):
        data_path =  self.data_info.iloc[index, 0]
        mel_spec = torch.load(data_path)
        label = self.data_info.iloc[index, 2]
        return mel_spec, label

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio classifier')
    parser.add_argument('--preprocess', action='store_true')
    args = parser.parse_args()

    DATA_DIR = 'preprocessed_data'
    INFO_FILE = os.path.join(DATA_DIR, 'data_info.csv')
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.preprocess == True:
        data_preprocessor = Preprocessor()
        data_preprocessor.preprocess_dir("data", DATA_DIR, INFO_FILE)

    data = AudioDataset(INFO_FILE, DATA_DIR)
    # print(f"there are {data.get_dataset_len()} samples in the dataset")

    signal, label = data[0]
    # print(signal[0].size())

    lstm = LSTM()
    predictions = lstm(signal.transpose(0,1))
    # print(predictions)


