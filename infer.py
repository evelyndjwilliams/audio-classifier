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
import numpy as np
import gc
from train import AudioDataset
from torchmetrics import ConfusionMatrix
from sklearn.metrics import confusion_matrix
import seaborn as sn
# import train 

def predict(model, data_loader, loss_fn, device, mode):
    running_loss = 0.0
    batch = 0
    true_classes = []
    predicted_classes = []
    with open('predictions.csv', 'w+') as o:
        for input, label in data_loader:
            true_classes.extend(label.tolist())
            one_hot = np.zeros((label.size()[0], 3))
            rows = np.arange(len(label))
            one_hot[rows, label] = 1
            target = torch.Tensor(one_hot).to(device).squeeze(0)
            prediction = model(input).to(device)
            print(prediction)
            o.write(f'{label},{torch.argmax(prediction, dim=1)}\n')
            predicted_classes.extend(torch.argmax(prediction, dim=1).tolist())
            loss = loss_fn(prediction, target)
            running_loss += loss.item()
            batch +=1
        print(f'avg {mode} loss = {running_loss/batch}')
    print(true_classes)#to_list())
    print(predicted_classes)#.to_list())
    # for i in

    target = torch.tensor(true_classes)
    preds = torch.tensor(predicted_classes)
    # confmat = ConfusionMatrix(num_classes=3)
    cm = confusion_matrix(preds, target)
    # cm = cm / cm.astype(np.float).sum(axis=1)
    print(cm)
    # classes = 
    classes = ['Speech', 'Rap', 'Singing']
    # (y_true, y_pred)
    df_cm = pd.DataFrame(cm, index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig('output.png')
    plt.matshow()
    return running_loss/batch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Audio classifier')
    parser.add_argument('--preprocess', action='store_true')
    parser.add_argument('--cnn', action='store_true')
    parser.add_argument('--lstm', action='store_true')
    args = parser.parse_args()

    BATCH_SIZE = 24
    DATA_DIR = 'preprocessed_data'
    INFO_FILE = os.path.join(DATA_DIR, 'data_info.csv')

    test_indices = pd.read_csv("test_indices.csv", header=None)
    test_indices = list(test_indices.iloc[:, 0])#.values.tolist())
    print(test_indices)
    """Create, shuffle & split training sets"""

    data = AudioDataset(INFO_FILE, DATA_DIR)    


    test_sampler = SubsetRandomSampler(test_indices)

    test_loader = DataLoader(data,
                                    batch_size=BATCH_SIZE,
                                    sampler=test_sampler,
                                    drop_last=False,
                                    num_workers=4)
    
    """Infer models"""

    LOSS_FN = nn.CrossEntropyLoss()
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.lstm== True:
        model = LSTM()
        model_name = 'lstm'

    if args.cnn== True:
        model = CNN()
        model_name = 'cnn'
        model.load_state_dict(torch.load('cnn_best_val_loss.pth'))

    model.eval()
    predict(model, test_loader, LOSS_FN, DEVICE, 'eval')


