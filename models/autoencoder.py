import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, input_shape, input_channels, param_size=0):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, encoder_channels[0], kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(encoder_channels[0], encoder_channels[1], kernel_size=3, padding=1), nn.ReLU()
        )

        self.bottleneck_dim = encoder_channels[-1]

        self.decoder = nn.Sequential(
            nn.Conv2d(self.bottleneck_dim, decoder_channels[0], kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv2d(decoder_channels[0], 1, kernel_size=1)
        )

        self.param_predictor = None
        if param_size > 0:
            self.param_predictor = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(self.bottleneck_dim, param_size)
            )

    def forward(self, x):
        z = self.encoder(x)
        recon = self.decoder(z)
        kernel_params = self.param_predictor(z) if self.param_predictor else None
        return recon, kernel_params