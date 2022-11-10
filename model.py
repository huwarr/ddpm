import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerSinusoidalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len = 1000):
        super().__init__()
        
        pos_s = torch.arange(max_len)
        i_s = torch.arange(embed_dim)

        sin_s = torch.sin(pos_s.unsqueeze(1) / 10000 ** (i_s / embed_dim)) * (i_s % 2 == 0)
        cos_s = torch.cos(pos_s.unsqueeze(1) / 10000 ** ((i_s - 1) / embed_dim)) * (i_s % 2 != 0)

        enc = sin_s + cos_s
        # (max_len X embed_dim)
        self.register_buffer('enc', enc)

    def forward(self, t):
        # t: (batch size X 1)
        return self.enc[t, :]


class ConvResidBlock(nn.Module):
    def __init__(self, n_groups, in_channels, out_channels, dropout=0.5, T=1000):
        super().__init__()

        self.basic_block = nn.Sequential(
            # DDPM, Appendix B -> "replaced weight normalization [49] with group normalization [66]"
            # Wide ResNet, 2 -> changes order from Conv->Norm->Activation to Norm->Activation->Conv
            nn.GroupNorm(num_groups=n_groups, num_channels=in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=n_groups, num_channels=out_channels),
            nn.ReLU(),
            # Wide Resnet, 2.4 -> "add a dropout layer into each residual block between convolutions and after ReLU"
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        # for residual connection
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)

        self.t_encoding = TransformerSinusoidalEncoding(embed_dim=in_channels, max_len=1000)

    def forward(self, x, t):
        # x: (batch size X channles X H x W)
        # t: (1)
        #t = t.expand(x.shape[0], 1)
        # t_encoded: (batch size X channels)
        t_encoded = self.t_encoding(t)
        t_encoded = t_encoded.view(t_encoded.shape[0], t_encoded.shape[1], 1, 1)
        x = x + t_encoded

        out = self.basic_block(x) + self.conv1(x)

        return out
    
    
class DownBlock(nn.Module):
    def __init__(self, n_groups, in_channels, out_channels, dropout=0.5, T=1000, self_attend=False):
        super().__init__()

        # DDPM, Appendix B -> "two convolutional residual blocks per resolution level"
        self.block_1 = ConvResidBlock(n_groups, in_channels, out_channels, dropout, T)
        self.block_2 = ConvResidBlock(n_groups, out_channels, out_channels, dropout, T)
        
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
        features = self.block_1(x, t)
        # features: (batch size X channels X H X W)
        if self.attention is not None:
            features_transformed = features.permute(0, 2, 3, 1)
            features_transformed = features_transformed.view(features_transformed.shape[0], -1, features_transformed.shape[-1])
            Q = self.q(features_transformed)
            K = self.k(features_transformed)
            V = self.v(features_transformed)
            features = self.attention(Q, K, V)[0].reshape(*features.shape)
        
        features = self.block_2(features, t)
        # Downsample
        x = self.downsample(features)
        # return output and features before downsampling to perform skip connections
        return x, features


class UpBlock(nn.Module):
    def __init__(self, n_groups, in_channels, out_channels, dropout=0.5, T=1000, self_attend=False):
        super().__init__()

        self.block_1 = ConvResidBlock(n_groups, in_channels * 2, out_channels, dropout, T)
        self.block_2 = ConvResidBlock(n_groups, out_channels, out_channels, dropout, T)
        
        if self_attend:
            self.q = nn.Linear(in_features=out_channels, out_features=out_channels)
            self.k = nn.Linear(in_features=out_channels, out_features=out_channels)
            self.v = nn.Linear(in_features=out_channels, out_features=out_channels)
            self.attention = nn.MultiheadAttention(embed_dim=out_channels, num_heads=1, batch_first=True)
        else:
            self.attention = None
        
        self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, padding=1, stride=2)

    def forward(self, x, skip_input, t):
        # Upsample
        x = self.upsample(x)
        # Skip connection
        x = torch.cat((skip_input, x), dim=1)

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
    def __init__(self, n_groups=32, in_channels=1, hid_chahhels=32, dropout=0.5, T=1000):
        super().__init__()
        # DDPM, Appendix B -> "32 × 32 models use four feature map resolutions (32 × 32 to 4 × 4)"
        self.down_blocks = []
        self.up_blocks = []
        self.levels_num = 4
        
        self.in_block = nn.Conv2d(in_channels, hid_chahhels, 1)
        self.out_block = nn.Conv2d(hid_chahhels, in_channels, 1)

        cur_channels = hid_chahhels
        for i in range(self.levels_num):
            self.down_blocks.append(DownBlock(n_groups, cur_channels, cur_channels * 2, dropout, T, i==1))
            self.up_blocks.append(UpBlock(n_groups, cur_channels * 2, cur_channels, dropout, T, i==1))
            cur_channels *= 2
        self.up_blocks.reverse()
        
        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.up_blocks = nn.ModuleList(self.up_blocks)

    def forward(self, x, t):
        x = self.in_block(x)
        skip_inputs = []

        for i, block in enumerate(self.down_blocks):
            x, features = block(x, t)
            skip_inputs.append(torch.clone(features))
        skip_inputs.reverse()

        for i, block in enumerate(self.up_blocks):
            x = block(x, skip_inputs[i], t)      
        
        return self.out_block(x)