# From deepsphere implementation at https://github.com/deepsphere/deepsphere-pytorch
# Modified to include SpatialConcreteDropout

# TODO: Double check this implementation

import healpy as hp
import torch
from torch import nn
from torch.nn import functional as F
import torch.utils.checkpoint as cp

from .chebyshev import SphericalChebConv
from .laplacian import get_laplacians
from .utils import HealpixBatchNorm, HealpixDownsample, HealpixUpsample
from .dropout import SpatialConcreteDropout

class ConcreteDropoutChebConv(nn.Module):
    def __init__(self, in_channels, out_channels, lap, kernel_size, weight_regularizer=1e-6, dropout_regularizer=1e-5):
        super().__init__()

        self.chebconv = SphericalChebConv(in_channels, out_channels, lap, kernel_size)
        self.dropout = SpatialConcreteDropout(weight_regularizer=weight_regularizer, dropout_regularizer=dropout_regularizer)
    
    def forward(self, lap, x):
        x = self.dropout(lap, x, self.chebconv)
        return x
class SphericalConvBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, kernel_size=3, weight_regularizer=1e-6, dropout_regularizer=1e-5,  gradient_checkpointing=False, **kwargs):
        super().__init__()

        self.in_channels = in_channels
        self.middle_channels = middle_channels
        self.out_channels = out_channels
        self.weight_regularizer = weight_regularizer
        self.dropout_regularizer = dropout_regularizer
        
        self.cheb1 = ConcreteDropoutChebConv(in_channels,
                                             middle_channels, 
                                             lap=None, 
                                             kernel_size=kernel_size, 
                                             weight_regularizer=weight_regularizer,
                                             dropout_regularizer=dropout_regularizer, 
                                             **kwargs)
        self.bn_r1 = nn.Sequential(
            HealpixBatchNorm(middle_channels),
            nn.ReLU()
        )
        self.cheb2 = ConcreteDropoutChebConv(middle_channels, 
                                             out_channels, 
                                             lap=None, 
                                             kernel_size=kernel_size,
                                             weight_regularizer=weight_regularizer,
                                             dropout_regularizer=dropout_regularizer,  
                                             **kwargs)
        self.bn_r2 = nn.Sequential(
            HealpixBatchNorm(out_channels),
            nn.ReLU()
        )

        self.gradient_checkpointing = gradient_checkpointing

    def forward_body(self, laplacian, x):
        """Forward pass of the spherical convolution block.

        Args:
            laplacian (torch.Tensor): Laplacian matrix.
            x (torch.Tensor): Input tensor of shape [batch x vertices x channels].

        Returns:
            torch.Tensor: Output tensor after applying the spherical convolution block.
        """
        x = self.cheb1(laplacian, x)
        x = self.bn_r1(x)
        x = self.cheb2(laplacian, x)
        x = self.bn_r2(x)

        return x
    
    def create_custom_forward(self, module):
                def custom_forward(*inputs):
                    return module(*inputs)
                return custom_forward

    def forward(self, laplacian, x):
        """Forward pass of the spherical convolution block with optional gradient checkpointing.

        Args:
            laplacian (torch.Tensor): Laplacian matrix.
            x (torch.Tensor): Input tensor of shape [batch x vertices x channels].

        Returns:
            torch.Tensor: Output tensor after applying the spherical convolution block.
        """
        if self.gradient_checkpointing and self.training:

            
            
            return cp.checkpoint(self.create_custom_forward(self.forward_body), laplacian, x, use_reentrant=False)
        else:
            return self.forward_body(laplacian, x)
    
def ensure_depth(N, depth):
    """Ensure possibility of depth from nodes
    Args:
        N (int): Number of nodes
        depth (int): Depth of the model
    """
    if N < 2**depth:
        raise ValueError(f"Depth {depth} is too large for N={N}.")
    if depth < 1:
        raise ValueError(f"Depth {depth} is too small.")
    return depth


# Does bottleneck count as part of the depth?
# If so, then depth = 3 + 1
# If not, then depth = 3
DEPTH = 3
# Specify middle and output channels for each block
ENCODER_CHANNELS = [[64, 64], [128, 128], [256, 256]]
BOTTLENECK_CHANNELS = [512, 256]
DECODER_CHANNELS = [[256, 128], [128, 64], [64, 64]]


class BayesianSphericalUNet(nn.Module):
    def __init__(self,
                 nside,
                 input_channels=9,
                 depth=DEPTH, 
                 encoder_channels=ENCODER_CHANNELS, 
                 bottleneck_channels=BOTTLENECK_CHANNELS, 
                 decoder_channels=DECODER_CHANNELS,
                 laps=None, 
                 kernel_size=3,
                 laplacian_type='combinatorial',
                 weight_regularizer=1e-6,
                 dropout_regularizer=1e-5):
        super().__init__()

        self.nside = nside
        self.npix = hp.nside2npix(nside)
        self.depth = ensure_depth(self.npix, depth)
        
        if laps is not None:
            laps = torch.load(laps)
        else:
            laps = get_laplacians(self.npix, self.depth + 1, laplacian_type)
        self.laps = []
        for i, lap in enumerate(laps):
            self.register_buffer(f'laplacian_{i}', lap)
            self.laps.append(getattr(self, f'laplacian_{i}'))

        if encoder_channels is not None:
            self.encoder_channels = encoder_channels
            assert len(encoder_channels) == self.depth, "Encoder channels must match the depth of the model."
        else:
            self.encoder_channels = [[64 * (2**i), 64 * (2**i)] for i in range(self.depth)]
        if bottleneck_channels is not None:
            self.bottleneck_channels = bottleneck_channels
            assert len(bottleneck_channels) == 2, "Bottleneck channels must be of length 2."
        else:
            self.bottleneck_channels = [512, 256]
        if decoder_channels is not None:
            self.decoder_channels = decoder_channels
            assert len(decoder_channels) == self.depth, "Decoder channels must match the depth of the model."
        else:
            self.decoder_channels = [[64 * (2**i), 64 * (2**i)] for i in range(self.depth, -1, -1)]
        

        self.encoder_block = nn.ModuleList([
            SphericalConvBlock(input_channels if i == 0 else self.encoder_channels[i-1][1],
                               self.encoder_channels[i][0],
                               self.encoder_channels[i][1],
                               kernel_size=kernel_size,
                               weight_regularizer=weight_regularizer,
                               dropout_regularizer=dropout_regularizer,
                               )
            for i in range(self.depth)
        ])
        self.bottleneck_block = SphericalConvBlock(self.encoder_channels[-1][1],
                                                   self.bottleneck_channels[0],
                                                   self.bottleneck_channels[1],
                                                   kernel_size=kernel_size,
                                                   weight_regularizer=weight_regularizer,
                                                   dropout_regularizer=dropout_regularizer,
                                                   )

        # bottleneck is 256
        # decoder would be (256 + 256, 256, 128) -> (128 + 128, 128, 64) -> (64 + 64, 64, 64)
        self.decoder_block = nn.ModuleList([
            SphericalConvBlock(self.decoder_channels[i][0] * 2,
                               self.decoder_channels[i][0],
                               self.decoder_channels[i][1],
                               kernel_size=kernel_size,
                               weight_regularizer=weight_regularizer,
                               dropout_regularizer=dropout_regularizer,
                               )
            for i in range(self.depth)
        ])

        self.pool = HealpixDownsample(kernel_size=4)
        self.upsample = HealpixUpsample(scale_factor=4)

        self.mu_conv = nn.Conv1d(self.decoder_channels[-1][1], 1, kernel_size=1)
        self.logvar_conv = nn.Conv1d(self.decoder_channels[-1][1], 1, kernel_size=1)
        self.logvar_conv.weight.data.normal_(0, 1e-6)
        self.logvar_conv.bias.data.fill_(0)
        self.cd_mu = SpatialConcreteDropout(channels_first=True, weight_regularizer=weight_regularizer, dropout_regularizer=dropout_regularizer)
        self.cd_logvar = SpatialConcreteDropout(channels_first=True, weight_regularizer=weight_regularizer, dropout_regularizer=dropout_regularizer)

        self.gradient_checkpointing = False

    def set_gradient_checkpointing(self, value):
        """Set gradient checkpointing for the model.

        Args:
            value (bool): Whether to enable or disable gradient checkpointing.
        """
        self.gradient_checkpointing = value
        for i, block in enumerate(self.encoder_block):
            block.gradient_checkpointing = value
        self.bottleneck_block.gradient_checkpointing = value
        for i, block in enumerate(self.decoder_block):
            block.gradient_checkpointing = value
    

    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for the model."""
        self.set_gradient_checkpointing(True)
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing for the model."""
        self.set_gradient_checkpointing(False)
    
    def is_gradient_checkpointing_enabled(self):
        return any(hasattr(block, 'gradient_checkpointing') and block.gradient_checkpointing for block in self.modules())
    
    def forward(self, x):
        """Forward pass of the Spherical UNet.

        Args:
            x (torch.Tensor): Input tensor of shape [batch x vertices x channels].

        Returns:
            torch.Tensor: Output tensor after applying the Spherical UNet.
        """

        laps = [getattr(self, f'laplacian_{i}') for i in range(len(self.laps))]
        # Encoder
        enc_features = []
        for i, block in enumerate(self.encoder_block):
            x = block(laps[-(i+1)], x)
            enc_features.append(x)
            x = self.pool(x)
            

        # Bottleneck
        x = self.bottleneck_block(laps[0], x)
        
        # Decoder
        for i, block in enumerate(self.decoder_block):
            x = self.upsample(x)
            x = torch.cat([x, enc_features.pop()], dim=2)
            x = block(laps[(i+1)], x)

        # Final convolution
        x = x.permute(0, 2, 1)
        mu = self.cd_mu(lap=None, x=x, layer=self.mu_conv)
        logvar = self.cd_logvar(lap=None, x=x, layer=self.logvar_conv)

        return mu, logvar
