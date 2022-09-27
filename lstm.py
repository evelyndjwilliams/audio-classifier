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
        # self.softmax = nn.Softmax(dim=1) # not using softmax because CrossEntropyLoss applies log softmax


    def forward(self, input_data):
        x = self.lstm(input_data)
        # print(f"after lstm part 0: {x[0].size()}")
        # print(f"after lstm part 1 0 : {x[1][0].size()}")
        # print(f"after lstm part 1 1 : {x[1][1].size()}")
        # this is using final hidden state (not cell state) as input to linear
        # as here https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html?highlight=lstm#torch.nn.LSTM
        x = self.linear1(x[1][0])
        # print(f"after linear 1: {x[0].size()}")
        x = self.relu(x)
        # print(f"after relu: {x[0].size()}")
        x = self.dropout(x)
        # print(f"after dropout: {x[0].size()}")
        x = self.linear2(x)
        # print(f"after linear 2: {x[0].size()}")
        predictions = x
        return predictions.squeeze(0)




    # data = AudioDataset(INFO_FILE, DATA_DIR)
    # # print(f"there are {data.get_dataset_len()} samples in the dataset")

    # signal, label = data[0]
    # # print(signal[0].size())

    # lstm = LSTM()
    # predictions = lstm(signal.transpose(0,1))
    # # print(predictions)


