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
# from lstm import AudioDataset
from sklearn.preprocessing import OneHotEncoder
from livelossplot import PlotLosses
import numpy as np
import gc

# def train_single_epoch(model, data_loader, loss_fn, optimiser, device, mode):
#     onehot_encoder = OneHotEncoder(sparse=False)
#     running_loss = 0.0
#     batch = 0
#     for input, target in data_loader:
#         optimiser.zero_grad()
#         one_hot = np.zeros((target.size()[0], 3))
#         rows = np.arange(len(target))
#         one_hot[rows, target] = 1
#         input, target = input.to(device), torch.Tensor(one_hot).to(device)#.unsqueeze(0).transpose(0,1)
#         prediction = model(input).transpose(0,1)
#         print(f'prediction size: {prediction.size()}, target size: {target.size()}')
#         loss = loss_fn(prediction.squeeze(1), target)
#         if mode == 'eval':
#             print(loss)
#             scheduler.step()
#         running_loss += loss.item()
#         batch +=1
#         if mode == 'train':
#             loss.backward()
#             optimiser.step()  
#             gc.collect()

#     print(f"loss: {loss.item()}")
#     print(f'avg {mode} loss = {running_loss/batch}')
#     return running_loss/batch

def train_single_epoch_cnn(model, data_loader, loss_fn, optimiser, device, mode):
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
        print(f'prediction size: {prediction.size()}, target size: {target.size()}')
        loss = loss_fn(prediction, target)
        print(loss.item())
        if mode == 'eval':
            scheduler.step()
        running_loss += loss.item()
        
        if mode == 'train':
            loss.backward()
            optimiser.step() 
            gc.collect()
        batch +=1
    print(f"loss: {loss.item()}")
    print(f'avg {mode} loss = {running_loss/batch}')
    return running_loss/batch

# def train(model, train_loader, val_loader, loss_fn, optimiser, device, epochs, log_file):
#     train_losses = []
#     val_losses = []
#     with open(log_file, 'w+') as o:
#         pass
#     for i in range(epochs):
#         # logs = {}
#         model.train()
#         print(f"Epoch {i+1}")
#         train_loss = train_single_epoch(model, train_loader, loss_fn, optimiser, device, 'train')
#         train_losses.append(train_loss)
#         model.eval()
#         val_loss = train_single_epoch(model, val_loader, loss_fn, optimiser, device, 'eval')
#         val_losses.append(val_loss)
#         print("---------------------------")
#         plt.plot(train_losses, linestyle = 'dotted', label='train')
        

#         # plt.plot(x1, y1, label = "line 1")
#         # line 2 points
#         # x2 = [10,20,30]
#         # y2 = [40,10,30]
#         # plotting the line 2 points 
#         plt.plot(val_losses, linestyle = 'dotted', label = "val")
#         plt.xlabel('Epoch')
#         # Set the y axis label of the current axis.
#         plt.ylabel('Cross Entropy Loss')
#         # Set a title of the current axes.
#         plt.title('Cross Entropy Loss computed on audio training and validation sets.')
#         # show a legend on the plot
#         if i == 0:
#             plt.legend()
#         plt.savefig(f'lstm training curve.png')
#         with open(log_file, 'a') as o:
#             o.write(f'Epoch {i+1}\n Train loss: {train_loss} \n Val loss: {val_loss}')
#     print("Finished training")

def train_cnn(model, train_loader, val_loader, loss_fn, optimiser, device, epochs, log_file):
    best_val_loss = 1000
    train_losses = []
    val_losses = []
    with open(log_file, 'w+') as o:
        pass
    for i in range(epochs):
        # logs = {}
        model.train()
        print(f"Epoch {i+1}")
        train_loss = train_single_epoch_cnn(model, train_loader, loss_fn, optimiser, device, 'train')
        train_losses.append(train_loss)
        model.eval()
        val_loss = train_single_epoch_cnn(model, val_loader, loss_fn, optimiser, device, 'eval')
        if val_loss <= best_val_loss:
            torch.save(model.state_dict(), f"best_val_loss_cnn_{i+1}.pth")
        val_losses.append(val_loss)
        print("---------------------------")
        plt.plot(train_losses, linestyle = 'dotted', label='train', color='green')
        plt.plot(val_losses, linestyle = 'dotted', label = "val", color='orange')
        plt.xlabel('Epoch')
        # Set the y axis label of the current axis.
        plt.ylabel('Cross Entropy Loss')
        # Set a title of the current axes.
        plt.title('Cross Entropy Loss computed on audio training and validation sets.')
        
        # show a legend on the plot
        if i == 0:
        # legend.remove()
            plt.legend()
        plt.savefig(f'cnn training curve.png')
        with open(log_file, 'a') as o:
            o.write(f'Epoch {i+1}\n Train loss: {train_loss} \n Val loss: {val_loss}\n')
        torch.save(model.state_dict(), f"{log_file[:5]}.pth")
    print("Finished training")

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
    parser.add_argument('--cnn', action='store_true')
    parser.add_argument('--lstm', action='store_true')
    args = parser.parse_args()

    DATA_DIR = 'preprocessed_data'
    INFO_FILE = os.path.join(DATA_DIR, 'data_info.csv')
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.preprocess == True:
        data_preprocessor = Preprocessor()
        data_preprocessor.preprocess_dir("data", DATA_DIR, INFO_FILE)

    # hyperparameters
    SAMPLE_RATE = 16000
    BATCH_SIZE = 24
    EPOCHS = 100
    LR = 0.0001
    
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
    train_indices, val_indices, test_indices = indices[test_split:], indices[:val_split], indices[val_split:test_split]
    with open ('test_indices.csv', 'w+') as o:
        for i in test_indices:
            o.write(f'{i} \n')
    """Creating Torch data samplers and loaders:"""
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    train_loader = DataLoader(data,
                                    batch_size=BATCH_SIZE,
                                    sampler=train_sampler,
                                    drop_last=True)
    val_batch_size = 1 if args.cnn == True else len(val_indices)   
    val_loader = DataLoader(data,
                                    batch_size=BATCH_SIZE,
                                    sampler=val_sampler,
                                    drop_last=True)
    val_set = Subset(data, val_indices)
    loader = DataLoader(data, batch_size=BATCH_SIZE)

    if args.lstm== True:
        lstm = LSTM()
        # optimiser = torch.optim.Adam(lstm.parameters(),lr=LR)
        # optimiser = optim.SGD(lstm.parameters(), lr=LR, momentum=0.9)
        optimiser = optim.SGD(lstm.parameters(), 0.1)
        scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.9)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser)
        train_cnn(lstm, train_loader, val_loader, LOSS_FN, optimiser, DEVICE, EPOCHS, 'lstm_log.txt')
        torch.save(lstm.state_dict(), "lstm.pth")
        print("Trained lstm saved at lstm.pth")

    if args.cnn== True:
        cnn = CNN()
        # optimiser = torch.optim.Adam(cnn.parameters(),lr=LR)
        # optimiser = optim.SGD(cnn.parameters(), lr=LR, momentum=0.9)
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimiser)
        optimiser = optim.SGD(cnn.parameters(), 0.1)
        scheduler = optim.lr_scheduler.ExponentialLR(optimiser, gamma=0.8)
        train_cnn(cnn, train_loader, val_loader, LOSS_FN, optimiser, DEVICE, EPOCHS, 'cnn_log.txt')
        torch.save(cnn.state_dict(), "cnn.pth")
        print("Trained cnn saved at cnn.pth")


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
# if CEL still bad --> look at loss : softmax, or chnge loss function

# CODE REFACTORING
# move hyperparams to dict file
# comment for readability
# exclude all LSTM stuff
# TRY MFCCs + deltas



