import math
import torch
from torch import nn
import torch.nn.functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, T, d_model, dim):
        assert d_model % 2 == 0
        super().__init__()
        emb = torch.arange(0, d_model, step=2) / d_model * math.log(10000)
        emb = torch.exp(-emb)
        pos = torch.arange(T).float()
        emb = pos[:, None] * emb[None, :]
        assert list(emb.shape) == [T, d_model // 2]
        emb = torch.stack([torch.sin(emb), torch.cos(emb)], dim=-1)
        assert list(emb.shape) == [T, d_model // 2, 2]
        emb = emb.view(T, d_model)

        self.timembedding = nn.Sequential(
            nn.Embedding.from_pretrained(emb),
            nn.Linear(d_model, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, t):
        emb = self.timembedding(t)
        return emb


class ConvResidBlock(nn.Module):
    """
    Convolutional residual block, based on Wide ResNet
    """
    def __init__(self, n_groups, in_channels, out_channels, time_embed_dim, dropout):
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

        self.basic_block_1 = nn.Sequential(
            # DDPM, Appendix B -> "replaced weight normalization [49] with group normalization [66]"
            # Wide ResNet, 2 -> changes order from Conv->Norm->Activation to Norm->Activation->Conv
            nn.GroupNorm(num_groups=n_groups, num_channels=in_channels),
            nn.SiLU(),
            # kernel_size=3 and padding=1 => image size will be unchanged
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        )
        self.basic_block_2 = nn.Sequential(
            nn.GroupNorm(num_groups=n_groups, num_channels=out_channels),
            nn.SiLU(),
            # Wide Resnet, 2.4 -> "add a dropout layer into each residual block between convolutions and after ReLU"
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, out_channels),
        )

        # for residual connection, we need to change number of channels
        if in_channels != out_channels:
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.conv1 = nn.Identity()
        # DDPM, Appendix B -> "Diffusion time t is specified by adding the 
        # Transformer sinusoidal position embedding [60] into each residual block"
        self.initialize()

    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.basic_block_2[-1].weight, gain=1e-5)

    def forward(self, x, t):
        out = self.basic_block_1(x) + self.time_emb_proj(t)[:, :, None, None]
        out = self.basic_block_2(out)
        # residual connection
        out += self.conv1(x)
        return out
    

class AttnBlock(nn.Module):
    def __init__(self, n_groups, in_channels):
        super().__init__()
        self.group_norm = nn.GroupNorm(n_groups, in_channels)
        self.proj_q = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_k = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj_v = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.proj = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.initialize()

    def initialize(self):
        for module in [self.proj_q, self.proj_k, self.proj_v, self.proj]:
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        nn.init.xavier_uniform_(self.proj.weight, gain=1e-5)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.group_norm(x)
        q = self.proj_q(h)
        k = self.proj_k(h)
        v = self.proj_v(h)

        q = q.permute(0, 2, 3, 1).view(B, H * W, C)
        k = k.view(B, C, H * W)
        w = torch.bmm(q, k) * (int(C) ** (-0.5))
        assert list(w.shape) == [B, H * W, H * W]
        w = F.softmax(w, dim=-1)

        v = v.permute(0, 2, 3, 1).view(B, H * W, C)
        h = torch.bmm(w, v)
        assert list(h.shape) == [B, H * W, C]
        h = h.view(B, H, W, C).permute(0, 3, 1, 2)
        h = self.proj(h)

        return x + h


class DownBlock(nn.Module):
    """
    UNet's downsampling block for single resolution level
    """
    def __init__(self, n_groups, in_channels, out_channels, time_embed_dim, dropout, self_attend=False, add_downsample=True):
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
        self.block_1 = ConvResidBlock(n_groups, in_channels, out_channels, time_embed_dim, dropout)
        self.block_2 = ConvResidBlock(n_groups, out_channels, out_channels, time_embed_dim, dropout)
        
        # DDPM, Appendix B -> "self-attention blocks at the 16 × 16 resolution between the convolutional blocks"
        if self_attend:
            self.attention_1 = AttnBlock(n_groups, out_channels)
            self.attention_2 = AttnBlock(n_groups, out_channels)
        else:
            self.attention_1 = nn.Identity()
            self.attention_2 = nn.Identity()
        
        # PixelCNN++, 2.3 -> "propose to use downsampling by using convolutions of stride 2"
        if add_downsample:
            self.downsample = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=2)
        else:
            self.downsample = nn.Identity()
        self.initialize()
        
    def initialize(self):
        if isinstance(self.downsample, nn.Conv2d):
            nn.init.xavier_uniform_(self.downsample.weight)
            nn.init.zeros_(self.downsample.bias)

    def forward(self, x, t):
        # First residual block
        features = self.block_1(x, t)
        features = self.attention_1(features)
        # Second residual block
        features = self.block_2(features, t)
        features = self.attention_2(features)
        # Downsampling
        x = self.downsample(features)
        # Return output and features before downsampling to perform skip connections
        return x, features


class UpBlock(nn.Module):
    """
    UNet's upsampling block for single resolution level
    """
    def __init__(self, n_groups, in_channels, out_channels, time_embed_dim, dropout, self_attend=False, add_upsample=True):
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
        self.block_1 = ConvResidBlock(n_groups, in_channels * 2, out_channels, time_embed_dim, dropout)
        self.block_2 = ConvResidBlock(n_groups, out_channels, out_channels, time_embed_dim, dropout)
        self.block_3 = ConvResidBlock(n_groups, out_channels, out_channels, time_embed_dim, dropout)
        if self_attend:
            self.attention_1 = AttnBlock(n_groups, out_channels)
            self.attention_2 = AttnBlock(n_groups, out_channels)
            self.attention_3 = AttnBlock(n_groups, out_channels)
        else:
            self.attention_1 = nn.Identity()
            self.attention_2 = nn.Identity()
            self.attention_3 = nn.Identity()
        
        # PixelCNN++ -> upsampling with transposed strided convolution
        # We need kernel_size=4 here to match sizes of upsampled image and features, passed with skip connection
        if add_upsample:
            self.upsample = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=4, padding=1, stride=2)
        else:
            self.upsample = nn.Identity()
        self.initialize()

    def initialize(self):
        if isinstance(self.upsample, nn.ConvTranspose2d):
            nn.init.xavier_uniform_(self.upsample.weight)
            nn.init.zeros_(self.upsample.bias)

    def forward(self, x, skip_input, t):
        # Upsample
        x = self.upsample(x)
        # Concatenate upsampled input and input from skip connection
        x = torch.cat((skip_input, x), dim=1)
        # Now apply upsampling block
        features = self.block_1(x, t)
        features = self.attention_1(features)

        features = self.block_2(x, t)
        features = self.attention_2(features)

        features = self.block_3(x, t)
        features = self.attention_3(features)

        return features


class MiddleBlock(nn.Module):

    def __init__(self, n_groups, n_channels, time_embed_dim, dropout):
        super().__init__()
        # UpSample block is applied to a concatenation of upsampled image and features, passed with skip connection.
        # Hence why, 2 times more input channels
        self.block_1 = ConvResidBlock(n_groups, n_channels, n_channels, time_embed_dim, dropout)
        self.block_2 = ConvResidBlock(n_groups, n_channels, n_channels, time_embed_dim, dropout)
            
        self.attention = AttnBlock(n_groups, n_channels)

    def forward(self, x, t):
        # Now apply upsampling block
        features = self.block_1(x, t)
        features = self.attention(features)

        features = self.block_2(x, t)

        return features


class UNet(nn.Module):
    """
    UNet - the model we will use to predict noise for each step
    """
    def __init__(self, n_groups=32, in_channels=3, hid_chahhels=128, dropout=0.1, T=1000):
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
        time_embed_dim = hid_chahhels * 4
        self.time_embedding = TimeEmbedding(T, hid_chahhels, time_embed_dim)
        # We need to change number of input channels to make Group norm work.
        # Precisely, it must be dividsble by n_groups
        assert hid_chahhels % n_groups == 0
        self.in_block = nn.Conv2d(in_channels, hid_chahhels, kernel_size=3, padding=1)
        
        # Each UNet's block increases/decreases number of channels twice
        cur_channels = hid_chahhels
        for i in range(self.levels_num):
            next_ch = cur_channels if i == 0 else cur_channels * 2
            self.down_blocks.append(DownBlock(n_groups, cur_channels, next_ch, time_embed_dim, dropout, i==1, add_downsample=(i==(self.levels_num - 1))))
            self.up_blocks.append(UpBlock(n_groups, next_ch, cur_channels, time_embed_dim, dropout, i==1, add_upsample=(i==(self.levels_num - 1))))
            cur_channels = next_ch
        self.up_blocks.reverse()
        # Middle blocks
        self.middle_block = MiddleBlock(n_groups, cur_channels, time_embed_dim, dropout)
        # Don't forget to return to the initial number of channels at the end
        self.out_block = nn.Sequential(
            nn.GroupNorm(n_groups, cur_channels),
            nn.SiLU(),
            nn.Conv2d(cur_channels, in_channels, kernel_size=3, padding=1)
        )
        # Module list to make gradients flow to each block as well
        self.down_blocks = nn.ModuleList(self.down_blocks)
        self.up_blocks = nn.ModuleList(self.up_blocks)

        self.initialize()
    
    def initialize(self):
        nn.init.xavier_uniform_(self.in_block.weight)
        nn.init.zeros_(self.in_block.bias)
        nn.init.xavier_uniform_(self.out_block[-1].weight, gain=1e-5)
        nn.init.zeros_(self.out_block[-1].bias)

    def forward(self, x, t):
        t = self.time_embedding(t)
        # Change number of channels of the image
        x = self.in_block(x)
        # List to store skip connections
        skip_inputs = []
        # Apply downsampling blocks
        for i, block in enumerate(self.down_blocks):
            x, features = block(x, t)
            skip_inputs.append(torch.clone(features))
        skip_inputs.reverse()
        # Middle block
        x = self.middle_block(x, t)
        # Now, upsampling
        for i, block in enumerate(self.up_blocks):
            x = block(x, skip_inputs[i], t)      
        # Return to the initial number of channels
        return self.out_block(x)


if __name__ == '__main__':
    batch_size = 8
    model = UNet(
        T=1000, ch=128, ch_mult=[1, 2, 2, 2], attn=[1],
        num_res_blocks=2, dropout=0.1)
    x = torch.randn(batch_size, 3, 32, 32)
    t = torch.randint(1000, (batch_size, ))
    y = model(x, t)