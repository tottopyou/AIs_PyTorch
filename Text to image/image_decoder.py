import torch
import torch.nn as nn

# Define and initialize the image decoder
class ImageDecoder(nn.Module):
    def __init__(self, latent_dim, output_channels):
        super(ImageDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_channels = output_channels
        self.fc = nn.Linear(256, latent_dim)
        self.conv_transpose = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 1, 0),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, output_channels, 4, 2, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, latent_dim, 1, 1)
        x = self.conv_transpose(x)
        return x

# Example usage
latent_dim = 256
output_channels = 3  # RGB channels
image_decoder = ImageDecoder(latent_dim, output_channels)

# Save the image decoder to a file
torch.save(image_decoder, "pretrained_image_decoder.pth")
