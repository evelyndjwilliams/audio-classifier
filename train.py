import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from torch.utils.data.sampler import SubsetRandomSampler
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
from lstm import LSTM
from lstm import AudioDataset
from sklearn.preprocessing import OneHotEncoder
from livelossplot import PlotLosses
import numpy as np
import gc

def train_single_epoch(model, data_loader, loss_fn, optimiser, device, mode):
    onehot_encoder = OneHotEncoder(sparse=False)
    # labels = [[1],[2],[3]]
    # onehot_encoder.fit(labels)  
    running_loss = 0.0
    batch = 0
    for input, target in data_loader:
        one_hot = np.zeros((target.size()[0], 3))
        rows = np.arange(len(target))
        one_hot[rows, target] = 1
        input, target = input.to(device), torch.Tensor(one_hot).to(device)#.unsqueeze(0).transpose(0,1)
        prediction = model(input).transpose(0,1)
        loss = loss_fn(prediction.squeeze(1), target)
        if mode == 'eval':
            print(loss)
            optimiser.step()  
            scheduler.step(loss.item())
        running_loss += loss.item()
        batch +=1
        if mode == 'train':
            optimiser.zero_grad()
            loss.backward()
            gc.collect()

    print(f"loss: {loss.item()}")
    print(f'avg {mode} loss = {running_loss/batch}')
    return running_loss/batch


def train(model, train_loader, val_loader, loss_fn, optimiser, device, epochs):
    train_losses = []
    val_losses = []
    for i in range(epochs):
        # logs = {}
        model.train()
        print(f"Epoch {i+1}")
        train_loss = train_single_epoch(model, train_loader, loss_fn, optimiser, device, 'train')
        train_losses.append(train_loss)
        model.eval()
        val_loss = train_single_epoch(model, val_loader, loss_fn, optimiser, device, 'eval')
        val_losses.append(val_loss)
        print("---------------------------")
        plt.plot(train_losses, linestyle = 'dotted', label='train')
        

        # plt.plot(x1, y1, label = "line 1")
        # line 2 points
        # x2 = [10,20,30]
        # y2 = [40,10,30]
        # plotting the line 2 points 
        plt.plot(val_losses, linestyle = 'dotted', label = "val")
        plt.xlabel('Epoch')
        # Set the y axis label of the current axis.
        plt.ylabel('Cross Entropy Loss')
        # Set a title of the current axes.
        plt.title('Cross Entropy Loss computed on audio training and validation sets.')
        # show a legend on the plot
        if i == 0:
            plt.legend()
        plt.savefig('training curve.png')

    print("Finished training")

if __name__ == "__main__":
    SAMPLE_RATE = 16000
    BATCH_SIZE = 32
    EPOCHS = 100
    LR = 0.001
    
    DATA_DIR = 'preprocessed_data'
    INFO_FILE = os.path.join(DATA_DIR, 'data_info.csv')
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOSS_FN = nn.CrossEntropyLoss()
    

    data = AudioDataset(INFO_FILE, DATA_DIR)    
    val_amount = 0.1
    test_amount = 0.1
    dataset_size = len(data)
    indices = list(range(dataset_size))
    val_split = int(np.floor(val_amount * dataset_size))
    test_split = val_split + int(np.floor(test_amount * dataset_size))
    shuffle_dataset = True
    random_seed = 42
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    # print(indices)
    train_indices, val_indices, test_indices = indices[test_split:], indices[:val_split], indices[val_split:test_split]
    print(val_indices)
    """Creating Torch data samplers and loaders:"""
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(data,
                                    batch_size=BATCH_SIZE,
                                    sampler=train_sampler)
    val_loader = DataLoader(data,
                                    batch_size=len(val_indices),
                                    sampler=val_sampler)
    val_set = Subset(data, val_indices)
    loader = DataLoader(data, batch_size=BATCH_SIZE)

    # construct model and assign it to device
    lstm = LSTM()
    # print(lstm)
    # optimiser = torch.optim.Adam(lstm.parameters(),lr=LR)
    optimiser = optim.SGD(lstm.parameters(), lr=LR, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser)
    # predictions = lstm(signal.transpose(0,1))
    # print(predictions)

    # initialise loss funtion + optimiser

    # train model
    
    
    train(lstm, train_loader, val_loader, LOSS_FN, optimiser, DEVICE, EPOCHS)
    # train(lstm, val_loader, LOSS_FN, optimiser, DEVICE, EPOCHS, 'eval')

    # save model
    torch.save(lstm.state_dict(), "lstm.pth")
    print("Trained lstm saved at lstm.pth")


# TO DO

# TRAINING SETS
# randomise shuffle and split data : 80% train, 10% val, 10% test
# shuffle labels and data same way

# PLOTTING
# plot valation loss curve

# TRAINING METHOD
# set scheduler / annealing --> test
# set scheduler to step() using val loss
# set adaptive LR
#
# TRY CNN
# if CEL still bad --> look at loss : softmax, or 

# TRY MFCCs + deltas



