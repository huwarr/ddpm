import torch
import torch.nn as nn


class TransformerSinusoidalEncoding(nn.Module):
    """
    Transformer sinusoidal positional embedding for encoding timestamps
    """
    def __init__(self, embed_dim, max_len = 1000):
        """
        embed_dim: int
            dimesionality of positional embedding
        max_len: int
            maximum sequence length; in our case, maximum timestamp
        """
        super().__init__()
        
        pos_s = torch.arange(max_len)
        i_s = torch.arange(embed_dim)

        sin_s = torch.sin(pos_s.unsqueeze(1) / 10000 ** (i_s / embed_dim)) * (i_s % 2 == 0)
        cos_s = torch.cos(pos_s.unsqueeze(1) / 10000 ** ((i_s - 1) / embed_dim)) * (i_s % 2 != 0)

        # (max_len X embed_dim)
        enc = sin_s + cos_s
        # register_buffer for moving embeddings matrix to
        # the same device as the model
        self.register_buffer('enc', enc)

    def forward(self, t):
        """
        t: torch.Tensor
            a batch of timestamps
        """
        # t: (batch size,)
        return self.enc[t, :]


class ConvResidBlock(nn.Module):
    """
    Convolutional residual block, based on Wide ResNet
    """
    def __init__(self, n_groups, in_channels, out_channels, dropout, T):
        """
        n_groups: int
            a parameter for group norm
        in_channels: int
            number of input channels
        out_channels: int
            number of output channels
        dropout: float
            probability of dropout in Dropout layer
        T: int
            total number of steps in diffusion process
        """
        super().__init__()

        self.basic_block = nn.Sequential(
            # DDPM, Appendix B -> "replaced weight normalization [49] with group normalization [66]"
            # Wide ResNet, 2 -> changes order from Conv->Norm->Activation to Norm->Activation->Conv
            nn.GroupNorm(num_groups=n_groups, num_channels=in_channels),
            nn.ReLU(),
            # kernel_size=3 and padding=1 => image size will be unchanged
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=n_groups, num_channels=out_channels),
            nn.ReLU(),
            # Wide Resnet, 2.4 -> "add a dropout layer into each residual block between convolutions and after ReLU"
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        # for residual connection, we need to change number of channels
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        # DDPM, Appendix B -> "Diffusion time t is specified by adding the 
        # Transformer sinusoidal position embedding [60] into each residual block"
        self.t_encoding = TransformerSinusoidalEncoding(embed_dim=in_channels, max_len=T)

    def forward(self, x, t):
        t_encoded = self.t_encoding(t)
        # x: (batch size X channles X H x W)
        # t_encoded: (batch size X channels)
        # => t_encoded needs reshaping
        t_encoded = t_encoded.view(t_encoded.shape[0], t_encoded.shape[1], 1, 1)
        # add positional embedding
        x = x + t_encoded
        # Residual block:
        out = self.basic_block(x) + self.conv1(x)
        return out
    
    
class DownBlock(nn.Module):
    """
    UNet's downsampling block for single resolution level
    """
    def __init__(self, n_groups, in_channels, out_channels, dropout, T, self_attend=False):
        """
        n_groups: int
            a parameter for group norm
        in_channels: int
            number of input channels
        out_channels: int
            number of output channels
        dropout: float
            probability of dropout in Dropout layer
        T: int
            total number of steps in diffusion process
        self_attend: bool
            whether we need a self attention block
        """
        super().__init__()

        # DDPM, Appendix B -> "two convolutional residual blocks per resolution level"
        self.block_1 = ConvResidBlock(n_groups, in_channels, out_channels, dropout, T)
        self.block_2 = ConvResidBlock(n_groups, out_channels, out_channels, dropout, T)
        
        # DDPM, Appendix B -> "self-attention blocks at the 16 × 16 resolution between the convolutional blocks"
        if self_attend:
            self.q = nn.Linear(in_features=out_channels, out_features=out_channels)
            self.k = nn.Linear(in_features=out_channels, out_features=out_channels)
            self.v = nn.Linear(in_features=out_channels, out_features=out_channels)
            self.attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=1, batch_first=True)
        else:
            self.attention = None
        
        # PixelCNN++, 2.3 -> "propose to use downsampling by using convolutions of stride 2"
        self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2)

    def forward(self, x, t):
        # First residual block
        features = self.block_1(x, t)
        # Self attention between residual blocks if needed
        if self.attention is not None:
            # features: (batch size X channels X H X W)
            # => needs reshaping into (batch size X L X feature_dim=channels)
            features_transformed = features.permute(0, 2, 3, 1)
            features_transformed = features_transformed.view(features_transformed.shape[0], -1, features_transformed.shape[-1])
            # attention time; don't forget to reshape back
            Q = self.q(features_transformed)
            K = self.k(features_transformed)
            V = self.v(features_transformed)
            features = self.attention(Q, K, V)[0].reshape(*features.shape)
        # Second residual block
        features = self.block_2(features, t)
        # Downsampling
        x = self.downsample(features)
        # Return output and features before downsampling to perform skip connections
        return x, features


class UpBlock(nn.Module):
    """
    UNet's upsampling block for single resolution level
    """
    def __init__(self, n_groups, in_channels, out_channels, dropout, T, self_attend=False):
        """
        n_groups: int
            a parameter for group norm
        in_channels: int
            number of input channels
        out_channels: int
            number of output channels
        dropout: float
            probability of dropout in Dropout layer
        T: int
            total number of steps in diffusion process
        self_attend: bool
            whether we need a self attention block
        """
        super().__init__()
        # UpSample block is applied to a concatenation of upsampled image and features, passed with skip connection.
        # Hence why, 2 times more input channels
        self.block_1 = ConvResidBlock(n_groups, in_channels * 2, out_channels, dropout, T)
        self.block_2 = ConvResidBlock(n_groups, out_channels, out_channels, dropout, T)
        if self_attend:
            self.q = nn.Linear(in_features=out_channels, out_features=out_channels)
            self.k = nn.Linear(in_features=out_channels, out_features=out_channels)
            self.v = nn.Linear(in_features=out_channels, out_features=out_channels)
            self.attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=1, batch_first=True)
        else:
            self.attention = None
        
        # PixelCNN++ -> upsampling with transposed strided convolution
        # We need kernel_size=4 here to match sizes of upsampled image and features, passed with skip connection
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, padding=1, stride=2)

    def forward(self, x, skip_input, t):
        # Upsample
        x = self.upsample(x)
        # Concatenate upsampled input and input from skip connection
        x = torch.cat((skip_input, x), dim=1)
        # Now apply upsampling block
        features = self.block_1(x, t)
        if self.attention is not None:
            features_transformed = features.permute(0, 2, 3, 1)
            features_transformed = features_transformed.view(features_transformed.shape[0], -1, features_transformed.shape[-1])
            Q = self.q(features_transformed)
            K = self.k(features_transformed)
            V = self.v(features_transformed)
            features = self.attention(Q, K, V)[0].reshape(*features.shape)
        out = self.block_2(features, t)
        return out


class UNet(nn.Module):
    """
    UNet - the model we will use to predict noise for each step
    """
    def __init__(self, n_groups=32, in_channels=1, hid_chahhels=32, dropout=0.5, T=1000):
        """
        n_groups: int
            a parameter for group norm
        in_channels: int
            number of input channels
        hid_chahhels: int
            number of channels of the first input of the first residual block;
            must be dividable by n_groups
        dropout: float
            probability of dropout in Dropout layer
        T: int
            total number of steps in diffusion process
        """
        super().__init__()

        self.down_blocks = []
        self.up_blocks = []
        # DDPM, Appendix B -> "32 × 32 models use four feature map resolutions (32 × 32) to (4 × 4)"
        self.levels_num = 4
        # We need to change number of input channels to make Group norm work.
        # Precisely, it must be dividsble by n_groups
        assert hid_chahhels % n_groups == 0
        self.in_block = nn.Conv2d(in_channels, hid_chahhels, 1)
        # Don't forget to return to the initial number of channels at the end
        self.out_block = nn.Conv2d(hid_chahhels, in_channels, 1)
        
        # Each UNet's block increases/decreases number of channels twice
        cur_channels = hid_chahhels
        for i in range(self.levels_num):
            self.down_blocks.append(DownBlock(n_groups, cur_channels, cur_channels * 2, dropout, T, i==1))
            self.up_blocks.append(UpBlock(n_groups, cur_channels * 2, cur_channels, dropout, T, i==1))
            cur_channels *= 2
        self.up_blocks.reverse()
        # Module list to make gradients flow to each block as well
        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.up_blocks = nn.ModuleList(self.up_blocks)

    def forward(self, x, t):
        # Change number of channels of the image
        x = self.in_block(x)
        # List to store skip connections
        skip_inputs = []
        # Apply downsampling blocks
        for i, block in enumerate(self.down_blocks):
            x, features = block(x, t)
            skip_inputs.append(torch.clone(features))
        skip_inputs.reverse()
        # Now, upsampling
        for i, block in enumerate(self.up_blocks):
            x = block(x, skip_inputs[i], t)      
        # Return to the initial number of channels
        return self.out_block(x)