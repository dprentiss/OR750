import torch
import torch.nn as nn
import torch.nn.functional as F

num_meters = 1
input_dim = num_meters
hidden_dim = 32
batch_dim = 1000
output_dim = 1
num_layers = 2

class AmiTest(nn.Module):
    def __init__(self, input_dim, hidden_dim, batch_dim, output_dim=1,
                 num_layers=2):
        super(AmiTest, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.batch_dim = batch_dim
        self.num_layers = num_layers

        self.lstm = nn.LSTM(self.input_dim, self.hidden_dim, self.num_layers)

        self.linear = nn.Linear(self.hidden_dim, output_dim)

    def init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_dim, self.hidden_dim),
                torch.zeros(self.num, self.batch_dim, self.hidden_dim))

    def forward(self, input):
        lstm_out, self.hidden = self.lstm(input.view(len(input),
                                                     self.batch_dim, -1))

        y_pred = self.linear(lstm_out[-1].view(self.batch_dim, -1))
        return y_pred.view(-1)

model = AmiTest(input_dim, hidden_dim, batch_dim, output_dim, num_layers)
