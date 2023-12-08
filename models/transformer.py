import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(
        self,
        input_steps,
        output_steps,
        d_model,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
    ):
        super(Transformer, self).__init__()
        self.input_steps = input_steps
        self.output_steps = output_steps
        self.d_model = d_model

        # Transformer Layer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            batch_first=True,
        )

        self.encoder_input_fc = nn.Linear(1, d_model)
        self.decoder_input_fc = nn.Linear(1, d_model)
        self.output_fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x shape: [batch, input_steps, 1] (1 feature for univariate)

        # Encoding
        encoder_input = self.encoder_input_fc(x)
        encoder_output = self.transformer.encoder(encoder_input)

        # Decoding
        decoder_input = torch.zeros(x.size(0), self.output_steps, 1, device=x.device)
        decoder_input = self.decoder_input_fc(decoder_input)
        decoder_output = self.transformer.decoder(decoder_input, encoder_output)

        # Fully connected layer for output
        output = self.output_fc(decoder_output).squeeze(-1)

        return output
