import torch
from torch.utils.data import Dataset, DataLoader


class TimeSeriesDataset(Dataset):
    def __init__(self, data, input_steps, output_steps):
        self.data = data
        self.input_steps = input_steps
        self.output_steps = output_steps

    def __len__(self):
        return len(self.data) - self.input_steps - self.output_steps + 1

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.input_steps]
        y = self.data[
            idx + self.input_steps : idx + self.input_steps + self.output_steps
        ]

        # Convert to float32
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)

        return x, y


def get_dataloader(data, input_steps, output_steps, batch_size):
    dataset = TimeSeriesDataset(data, input_steps, output_steps)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader
