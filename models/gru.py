import torch
import torch.nn as nn


class GRU(nn.Module):
    def __init__(self, input_steps, output_steps, hidden_size):
        super(GRU, self).__init__()
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.hidden_size = hidden_size

        # GRU layer
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, batch_first=True)

        # Fully connected layer: output from GRU to the predicted values
        self.fc = nn.Linear(hidden_size, output_steps)

    def forward(self, x):
        
        # GRU forward pass
        output, _ = self.gru(x)
        output = output[:, -1, :]
        output = self.fc(output)

        return output
