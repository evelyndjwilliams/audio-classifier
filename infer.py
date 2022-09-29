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
from sklearn.metrics import f1_score

def predict(model, data_loader, loss_fn, device, mode):
    running_loss = 0.0
    batch = 0
    true_classes = []
    predicted_classes = []
    f_paths = []
    mel_specs = []
    classes = ['Speech', 'Rap', 'Singing']
    with open('predictions.csv', 'w+') as o:
        for input, label, f_path in data_loader:
            mel_specs.extend(input.tolist())
            true_classes.extend(label.tolist())
            one_hot = np.zeros((label.size()[0], 3))
            rows = np.arange(len(label))
            one_hot[rows, label] = 1
            target = torch.Tensor(one_hot).to(device).squeeze(0)
            prediction = model(input).to(device)
            o.write(f'{label},{torch.argmax(prediction, dim=1)}\n')
            predicted_classes.extend(torch.argmax(prediction, dim=1).tolist())
            loss = loss_fn(prediction, target)
            running_loss += loss.item()
            batch +=1
            f_paths.extend(f_path)

        print(f'avg {mode} loss = {running_loss/batch}')

    target = torch.tensor(true_classes)
    preds = torch.tensor(predicted_classes)

    """Save wrong-labelled file paths"""
    with open('Incorrect label predictions.txt', 'w+') as o:
        for i, label in enumerate(target):
            if label != preds[i]:
                o.write(f"{f_paths[i]}\n")
                plt.imshow(torch.flip(torch.Tensor(mel_specs[i]).transpose(0,1), dims=(0,)))
                plt.savefig(f"correct/{classes[label]}_{os.path.basename(f_paths[i])[:-4]}.png")

    with open('stats.txt', 'w+') as o:
        """Confusion matrix"""
        cm = confusion_matrix(target, preds)
        o.write(f'confusion matrix: {cm}\n')
        df_cm = pd.DataFrame(cm, index = [i for i in classes],
                            columns = [i for i in classes])
        sn.heatmap(df_cm, annot=True)
        plt.xlabel('Predicted label', fontsize=12)
        plt.ylabel('True label', fontsize=12)
        plt.savefig('output.png')

        """F1 score"""
        o.write("f1 scores")
        o.write(f"{classes}\n")
        o.write(f'{f1_score(target, preds, average=None)}\n')

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
    test_indices = list(test_indices.iloc[:, 0])
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


