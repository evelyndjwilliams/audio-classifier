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
from cnn import CNN
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import gc

def train_single_epoch(model, data_loader, loss_fn, optimiser, device, mode):
    onehot_encoder = OneHotEncoder(sparse=False)
    running_loss = 0.0
    batch = 0
    for input, target in data_loader:
        optimiser.zero_grad()
        one_hot = np.zeros((target.size()[0], 3))
        rows = np.arange(len(target))
        one_hot[rows, target] = 1
        target = one_hot
        target = torch.Tensor(target).to(device).squeeze(0)
        prediction = model(input).to(device)
        # print(f'prediction size: {prediction.size()}, target size: {target.size()}')
        loss = loss_fn(prediction, target)
        # print(loss.item())
        running_loss += loss.item()
        if mode == 'train':
            loss.backward()
            optimiser.step() 
            gc.collect()
        batch +=1
    # print(f"loss: {loss.item()}")
    print(f'avg {mode} loss = {running_loss/batch}')
    return running_loss/batch

def train(model, train_loader, val_loader, loss_fn, optimiser, device, epochs, log_file):
    best_val_loss = 1000
    train_losses = []
    val_losses = []
    with open(log_file, 'w+') as o:
        pass
    for i in range(epochs):
        print(f"Epoch {i+1}")
        model.train()
        train_loss = train_single_epoch(model, train_loader, loss_fn, optimiser, device, 'train')
        train_losses.append(train_loss)
        model.eval()
        val_loss = train_single_epoch(model, val_loader, loss_fn, optimiser, device, 'eval')
        """ store model with lowest loss on val set"""
        if val_loss < best_val_loss:
            torch.save(model.state_dict(), f"{log_file[:4]}_best_val_loss_epoch_{i+1}.pth")
            best_val_loss = val_loss
        val_losses.append(val_loss)
        """Update and save training curves"""
        plt.plot(train_losses, linestyle = 'dotted', label='train', color='green')
        plt.plot(val_losses, linestyle = 'dotted', label = "val", color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Cross Entropy Loss')
        plt.title('Cross Entropy Loss computed on audio training and validation sets.')
        if i == 0:
            plt.legend()
        plt.savefig(f'cnn training curve no dropout.png')
        with open(log_file, 'a') as o:
            o.write(f'Epoch {i+1}\n Train loss: {train_loss} \n Val loss: {val_loss}\n')
        scheduler.step()
        print("---------------------------")
    print("Finished training")

class Preprocessor():
    """ Extracts scaled log mel-spectrograms from audio dataset.
        Stores in /preprocessed-data 
        To run preprocessing, pass --preprocess arg to train.py"""

    def __init__(self):
        self.transform = torchaudio.transforms.MelSpectrogram(sample_rate=16000, n_fft=2048, hop_length=512, n_mels=80)
        self.labels = {'speech':0, 'rap':1, 'singing':2}

    def preprocess(self, filename):
        wav, sr = torchaudio.load(filename, normalize=True)
        ms = self.transform(wav)
        log_ms = librosa.amplitude_to_db(ms.squeeze(0))
        mm_scaler = MinMaxScaler()
        mm_scaler.fit(log_ms)
        scaled_log_ms = mm_scaler.transform(log_ms)
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
    parser.add_argument('--cnn', action='store_true')
    parser.add_argument('--lstm', action='store_true')
    args = parser.parse_args()

    """Hyperparameters"""

    SAMPLE_RATE = 16000
    BATCH_SIZE = 24
    EPOCHS = 100
    LR = 0.1
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    LOSS_FN = nn.CrossEntropyLoss()

    """Preprocess data (mel-spectrogram generation)"""

    DATA_DIR = 'preprocessed_data'
    INFO_FILE = os.path.join(DATA_DIR, 'data_info.csv')

    if args.preprocess == True:
        data_preprocessor = Preprocessor()
        data_preprocessor.preprocess_dir("data", DATA_DIR, INFO_FILE)

    """Create, shuffle & split training sets"""

    data = AudioDataset(INFO_FILE, DATA_DIR)    

    val_amount = 0.1
    test_amount = 0.1
    dataset_size = len(data)
    indices = list(range(dataset_size))
    val_split = int(np.floor(val_amount * dataset_size))
    test_split = val_split + int(np.floor(test_amount * dataset_size))

    random_seed = 42
    np.random.seed(random_seed)
    np.random.shuffle(indices)
    train_indices, val_indices, test_indices = indices[test_split:], indices[:val_split], indices[val_split:test_split]

    # storing indices of test set
    with open ('test_indices.csv', 'w+') as o:
        for i in test_indices:
            o.write(f'{i} \n')

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_loader = DataLoader(data,
                                    batch_size=BATCH_SIZE,
                                    sampler=train_sampler,
                                    drop_last=True,
                                    num_workers=4)
    
    val_loader = DataLoader(data,
                                    batch_size=BATCH_SIZE,
                                    sampler=val_sampler,
                                    drop_last=True,
                                    num_workers=4)
    val_set = Subset(data, val_indices)
    loader = DataLoader(data, batch_size=BATCH_SIZE)

    """Train models"""

    if args.lstm== True:
        model = LSTM()
        model_name = 'lstm'

    if args.cnn== True:
        model = CNN()
        model_name = 'cnn'

    optimiser = optim.SGD(model.parameters(), LR)
    scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.95)
    train(model, train_loader, val_loader, LOSS_FN, optimiser, DEVICE, EPOCHS, f'{model_name}_log.txt')
    torch.save(model.state_dict(), f"{model_name}.pth")
    print(f"Trained {model_name} saved at {model_name}.pth")


# TO DO

# URGENT 
# backup to gh

# TRAINING METHOD
# hypterparameter tuning: reduce LR : (0.1, 0.05, 0.01)
# change gamma lav for scheduler
# if val loss is bouncing around : reduce LR
# TRY CNN
# compare dropout / no dropout (backup loss curves to compare)

# CODE REFACTORING
# move hyperparams to dict file
# comment for readability
# exclude all LSTM stuff - just check it runs



