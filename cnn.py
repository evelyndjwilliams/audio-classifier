import torch
import torch.nn as nn

class CNN(nn.Module):
    """ Defines architecture and training procedure for model.
        Model is a 1-layer LSTM with 182 hidden units."""

    def __init__(self):
        """Model architecture"""
        super().__init__()
        self.cnn1 = torch.nn.Conv2d(1, 64, (5,5), stride=1, padding='same')
        self.maxpool1 = torch.nn.MaxPool2d((2,2), padding=0, stride=2,  return_indices=False, ceil_mode=False)
        self.cnn2 = torch.nn.Conv2d(64, 64, (5,5), stride=1, padding='same')
        self.maxpool2 = torch.nn.MaxPool2d((2,2), padding=0, stride=2, return_indices=False, ceil_mode=False)
        self.relu = nn.ReLU()
        self.linear1 = nn.Linear(29440,500)
        self.linear2 = nn.Linear(500,3)
        # self.linear3 = nn.Linear(64, 32)
        # self.linear4 = nn.Linear(32,3)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_data):
        """Training order"""
        x = self.cnn1(input_data.unsqueeze(1))
        x = self.relu(x)
        x = self.maxpool1(x)
        x = self.cnn2(x)
        x = self.relu(x)
        x = self.maxpool2(x)
        x = torch.flatten(x, start_dim=1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        # ------------------------------
        # # if applying FC layer to channels dim
        # x = x.transpose(1,3)
        # x = self.linear3(x)
        # x = self.relu(x)
        # x = self.dropout(x)
        # x = self.linear4(x)
        # # Take mean in both spatial dimensions
        # x = x.mean(dim=1)
        # x = x.mean(dim=1)
        #--------------------------------
        logits = x
        return logits