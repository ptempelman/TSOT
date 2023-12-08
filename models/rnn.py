import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_steps):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )
        self.fc = nn.Linear(hidden_size, output_steps)

    def forward(self, x):
        # x shape: [batch, input_steps, features]
        output, _ = self.rnn(x)
        # output shape: [batch, input_steps, hidden_size]
        output = self.fc(output[:, -1, :])
        # output shape: [batch, output_steps]
        return output


def train_model(model, train_loader, lr, num_epochs):
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    model = model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        for inputs, targets in train_loader:
            inputs = inputs.float().unsqueeze(-1)  # Add feature dimension
            targets = targets.float()
            inputs, targets = inputs.to(device), targets.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            if torch.isnan(loss.cpu()):
                break

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), 1.0
            )  # Gradient clipping to prevent exploding gradients
            optimizer.step()
            
        if torch.isnan(loss):
            # print("Training stopped due to nan loss at epoch:", epoch)
            break

        # print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    return model
