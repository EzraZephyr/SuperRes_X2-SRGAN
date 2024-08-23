import torch.nn.functional as F
from torch import nn
from torchvision.models import vgg19

class PerceptualLoss(nn.Module):
    def __init__(self):
        super(PerceptualLoss, self).__init__()
        vgg = vgg19(pretrained=True).features
        self.layers = nn.Sequential(*list(vgg.children())[:35]).eval()
        for param in self.layers.parameters():
            param.requires_grad = False
        # Loads the pre-trained VGG19 network and extracts the first 35 layers for perceptual loss computation
        # Freezes these layers to prevent them from being updated during training

    def forward(self, output, target):
        output_features = self.layers(output)
        target_features = self.layers(target)
        return F.mse_loss(output_features, target_features)
    # Extracts high-level features from both generated and real images and computes the loss

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.PReLU(),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
        )
        # Constructs a residual block

    def forward(self, x):
        return x + self.block(x)
        # Performs residual connection

class Generator(nn.Module):
    def __init__(self, num_residual_blocks=16):
        super(Generator, self).__init__()

        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )
        # Initial convolutional layer

        residual_block = []
        for _ in range(num_residual_blocks):
            residual_block.append(ResidualBlock(64))

        self.residual_blocks = nn.Sequential(*residual_block)
        # Creates multiple residual blocks in a loop and combines them into a sequential module
        # Used for deep feature extraction in the generator network

        self.upscale = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.PixelShuffle(2),
            nn.PReLU(),
        )
        # Maps the number of channels and dimensions of the image, increasing the number of channels
        # Then upscales the image, arranging every four channels into the pixel positions [0,0], [0,1], [1,0], [1,1]
        # Doubles the resolution

        self.final = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)
        # Final convolutional layer, mapping the output back to an RGB image

    def forward(self, x):
        x = self.initial(x)
        residual = x
        x = self.residual_blocks(x)
        x = self.upscale(residual + x)
        x = self.final(x)
        return x
        # Passes the image through the initial convolutional layer,
        # saving the result as `residual` for later use in residual connection
        # Processes the image through 16 residual blocks to enhance features,
        # then performs the final residual connection with the initial `residual`
        # Upscales the resolution and outputs the image
