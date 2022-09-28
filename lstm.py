import torch
import torch.nn as nn

class LSTM(nn.Module):
    """ Defines architecture and training procedure for model.
        Model is a 1-layer LSTM with 182 hidden units."""

    def __init__(self):
        """Model architecture"""
        super().__init__()
        self.lstm = nn.LSTM(input_size=80, hidden_size=182, num_layers=1, batch_first=True)
        self.linear1 = nn.Linear(182,182)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.25)
        self.linear2 = nn.Linear(182,3)


    def forward(self, input_data):
        x = self.lstm(input_data)
        x = self.linear1(x[1][0])
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        logits = x
        return logits.squeeze(0)



