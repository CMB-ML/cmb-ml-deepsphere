from pathlib import Path
import math

import healpy as hp
import numpy as np
from scipy import sparse
import torch
from torch import nn

from deepsphere_unet.deepsphere.laplacian import (
    healpix_graph,
    prepare_laplacian,
    scipy_csr_to_sparse_tensor
)
from deepsphere_unet.deepsphere.dropout import SpatialConcreteDropout


LAP_DIR = Path("lap/")
LAP_DIR.mkdir(exist_ok=True, parents=True)


ALL_NSIDES = [2**i for i in range(0, 10)]

for nside in ALL_NSIDES:
    lap_fp = LAP_DIR / f"lap_{nside}.npz"
    if lap_fp.exists():
        print(f"Skipping {lap_fp}")
        continue
    print(f"Processing {lap_fp}")
    G = healpix_graph(nside, lap_type='combinatorial')
    G.compute_laplacian()
    some_lap = prepare_laplacian(G.L)
    sparse.save_npz(lap_fp, some_lap)

# Hardcoding for clarity in parameterizations
KERNEL_SIZE = 3
NCH = 9


def get_laplacian(nside):
    """Get the Laplacian matrix for a given nside."""
    lap_fp = LAP_DIR / f"lap_{nside}.npz"
    if not lap_fp.exists():
        raise FileNotFoundError(f"Laplacian file not found for nside {nside}")
    lap = sparse.load_npz(lap_fp)
    lap = scipy_csr_to_sparse_tensor(lap)
    return lap


def cheb_conv(laplacian, inputs, weight):
    """Chebyshev convolution.

    Args:
        laplacian (:obj:`torch.sparse.Tensor`): The laplacian corresponding to the current sampling of the sphere.
        inputs (:obj:`torch.Tensor`): The current input data being forwarded.
        weight (:obj:`torch.Tensor`): The weights of the current layer.

    Returns:
        :obj:`torch.Tensor`: Inputs after applying Chebyshev convolution.
    """

    # ORIGINAL
    # B, V, Fin = inputs.shape
    # K, Fin, Fout = weight.shape
    # B = batch size
    # V = number of vertices
    # K = order of Chebyshev polynomials
    # Fout = number of output features
    # Fin = number of input features

    # RENAMED ORIGINAL        
    B, V, P = inputs.shape                                   # B  x V  x P
    K, P, Q = weight.shape                                   # K  x P  x Q
    # B = batch size
    # V = number of vertices (=Npix)
    # K = order of Chebyshev polynomials
    # Q = number of output features
    # P = number of input features

    # REVISED (proposed)
    # V, V    = laplacian.shape                                # V  x V
    # Q, P, K = weight.shape                                   # Q  x P  x K
    # B, P, V = inputs.shape                                   # B  x P  x V
    # V = number of vertices (=Npix)
    # Q = number of output features
    # P = number of input features
    # K = order of Chebyshev polynomials
    # B = batch size

    # transform to Chebyshev basis
    x0 = inputs.permute(1, 2, 0).contiguous()                # V  x P   x B
    # x0 = inputs.permute(2, 1, 0).contiguous()                # V  x P   x B # proposed
    x0 = x0.view([V, P * B])                                 # V  x PB
    inputs = x0.unsqueeze(0)                                 # 1  x V  x PB

    if K > 0:
        x1 = torch.sparse.mm(laplacian, x0)                  # V  x PB
        inputs = torch.cat((inputs, x1.unsqueeze(0)), 0)     # 2  x V  x PB
        for _ in range(1, K - 1):
            x2 = 2 * torch.sparse.mm(laplacian, x1) - x0
            inputs = torch.cat((inputs, x2.unsqueeze(0)), 0) # M  x PB
            x0, x1 = x1, x2

    inputs = inputs.view([K, V, P, B])                       # K  x V  x P  x B
    inputs = inputs.permute(3, 1, 2, 0).contiguous()         # B  x V  x P  x K
    inputs = inputs.view([B * V, P * K])                     # BV x PK

    # Linearly compose Fin features to get Fout features
    # weight = weight.permute(2, 1, 0).contiguous()                         # K  x P  x Q  # proposed
    weight = weight.view(P * K, Q)                           # PK x Q
    inputs = inputs.matmul(weight)                           # BV x Q
    inputs = inputs.view([B, V, Q])                          # B  x V  x Q

    return inputs


def junk():
# def cheb_conv(laplacian, inputs, weight):
#     """Chebyshev convolution.

#     Args:
#         laplacian (:obj:`torch.sparse.Tensor`): The laplacian corresponding to the current sampling of the sphere.
#         inputs (:obj:`torch.Tensor`): The current input data being forwarded.
#         weight (:obj:`torch.Tensor`): The weights of the current layer.

#     Returns:
#         :obj:`torch.Tensor`: Inputs after applying Chebyshev convolution.
#     """

#     # ORIGINAL
#     B, V, Fin = inputs.shape
#     K, Fin, Fout = weight.shape
#     # B = batch size
#     # V = number of vertices
#     # K = order of Chebyshev polynomials
#     # Fout = number of output features
#     # Fin = number of input features

#     # RENAMED ORIGINAL        
#     B, V, P = inputs.shape                                   #      B  x V  x P
#     K, P, Q = weight.shape                                   # K  x P  x Q
#     # B = batch size
#     # V = number of vertices (=Npix)
#     # K = order of Chebyshev polynomials
#     # Q = number of output features
#     # P = number of input features

#     # REVISED
#     V, V    = laplacian.shape                                # V  x V
#     Q, P, K = weight.shape                                   # Q  x P  x K
#     B, P, V = inputs.shape                                   #      B  x V  x P
#     # V = number of vertices (=Npix)
#     # Q = number of output features
#     # P = number of input features
#     # K = order of Chebyshev polynomials
#     # B = batch size

#     # transform to Chebyshev basis
#     x0 = inputs.permute(1, 2, 0).contiguous()                #      V  x P   x B
#     x0 = x0.view([V, P * B])                                 #      V  x PB
#     inputs = x0.unsqueeze(0)                                 # 1  x V  x PB

#     if K > 0:
#         x1 = torch.sparse.mm(laplacian, x0)                  # V  x PB
#         inputs = torch.cat((inputs, x1.unsqueeze(0)), 0)     # 2  x V  x PB
#         for _ in range(1, K - 1):
#             x2 = 2 * torch.sparse.mm(laplacian, x1) - x0
#             inputs = torch.cat((inputs, x2.unsqueeze(0)), 0) # M  x PB
#             x0, x1 = x1, x2

#     inputs = inputs.view([K, V, P, B])                       # K  x V  x P  x B
#     inputs = inputs.permute(3, 1, 2, 0).contiguous()         # B  x V  x P  x K
#     inputs = inputs.view([B * V, P * K])                     # BV x PK

#     # Linearly compose Fin features to get Fout features
#     weight = weight.view(P * K, Q)                           # PK x Q
#     inputs = inputs.matmul(weight)                           # BV x Q
#     inputs = inputs.view([B, V, Q])                          # B  x V  x Q

#     return inputs
    pass


class ChebConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = KERNEL_SIZE
        # shape = (out_channels, in_channels, self.kernel_size)  # proposed
        shape = (self.kernel_size, in_channels, out_channels)
        self.weight = nn.Parameter(torch.Tensor(*shape))
        self.bias = nn.Parameter(torch.Tensor(out_channels))
        self.xavier_unif_initialization()  # Note that other options are possible

    def xavier_unif_initialization(self):
        """Initialize weights and bias.
        """
        std = math.sqrt(6 / (self.in_channels + self.out_channels))
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.fill_(0.01)

    def forward(self, lap, x):
        out = cheb_conv(lap, x, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out
    

class EncoderBlock(nn.Module):
    # Copying contents of BayesianSphericalChebBNPool2
    def __init__(self, lap, in_c, mid_c, out_c, pooling):
        super().__init__()
        self.lap = lap
        self.pooling = pooling

        self.conv1 = ChebConv(in_c, mid_c)
        self.d_o1 = SpatialConcreteDropout()
        self.bn1 = nn.BatchNorm1d(mid_c)
        self.relu1 = nn.ReLU()
        
        self.conv2 = ChebConv(mid_c, out_c)
        self.d_o2 = SpatialConcreteDropout()
        self.bn2 = nn.BatchNorm1d(out_c)
        self.relu2 = nn.ReLU()

    def forward(self, x):
        if self.pooling is not None:
            x = torch.permute(x, (0, 2, 1))
            x = self.pooling(x)
            x = torch.permute(x, (0, 2, 1))

        x = self.d_o1(self.lap, x, self.conv1)
        x = torch.permute(x, (0, 2, 1))
        x = self.bn1(x)
        x = torch.permute(x, (0, 2, 1))
        x = self.relu1(x)

        x = self.d_o2(self.lap, x, self.conv2)
        x = torch.permute(x, (0, 2, 1))
        x = self.bn2(x)
        x = torch.permute(x, (0, 2, 1))
        x = self.relu2(x)
        return x
    

class DecoderBlock(nn.Module):
    def __init__(self, lap, in_c, mid_c, out_c, unpooling):
        super().__init__()
        self.lap = lap
        self.unpooling = unpooling

        self.conv1 = ChebConv(in_c, mid_c)
        self.d_o1 = SpatialConcreteDropout()
        self.bn1 = nn.BatchNorm1d(mid_c)
        self.relu1 = nn.ReLU()

        self.conv2 = ChebConv(mid_c, out_c)
        self.d_o2 = SpatialConcreteDropout()
        self.bn2 = nn.BatchNorm1d(out_c)
        self.relu2 = nn.ReLU()

    def forward(self, x, concat_data):
        x = torch.permute(x, (0, 2, 1))
        x = self.unpooling(x)
        x = torch.permute(x, (0, 2, 1))
        x = torch.cat((x, concat_data), dim=2)

        x = self.d_o1(self.lap, x, self.conv1)
        x = torch.permute(x, (0, 2, 1))
        x = self.bn1(x)
        x = torch.permute(x, (0, 2, 1))
        x = self.relu1(x)

        x = self.d_o2(self.lap, x, self.conv2)
        x = torch.permute(x, (0, 2, 1))
        x = self.bn2(x)
        x = torch.permute(x, (0, 2, 1))
        x = self.relu2(x)
        return x


class DeepSphereUNetCD(nn.Module):
    def __init__(self, 
                #  nside
                 ):
        """
        Makes fixed UNet for testing.

        Encoders           Decoders
        E3  --------------  D3
          E2  ----------  D2
            E1  ------  D1
                  B
        """
        super().__init__()

        # self.nside = nside
        # self.npix = hp.nside2npix(nside)
        # self.kernel_size = 3  # Coded as a global

        self.L3_lap = get_laplacian(32)  # Hard-coding Nsides for clarity
        self.L2_lap = get_laplacian(16)
        self.L1_lap = get_laplacian(8)
        self.L0_lap = get_laplacian(4)

        pool = nn.MaxPool1d(kernel_size=4)
        unpool = nn.Upsample(scale_factor=4, mode='nearest')

        # Encoder
        self.E3 = EncoderBlock(self.L3_lap, in_c=NCH, mid_c= 32, out_c= 64, pooling=None)
        self.E2 = EncoderBlock(self.L2_lap, in_c= 64, mid_c=128, out_c=128, pooling=pool)
        self.E1 = EncoderBlock(self.L1_lap, in_c=128, mid_c=256, out_c=256, pooling=pool)

        # Bottleneck
        self.B  = EncoderBlock(self.L0_lap, in_c=256, mid_c=512, out_c=256, pooling=pool)

        # Decoder
        self.D1 = DecoderBlock(self.L1_lap, in_c=256+256, mid_c=256, out_c=128, unpooling=unpool)
        self.D2 = DecoderBlock(self.L2_lap, in_c=128+128, mid_c=128, out_c= 64, unpooling=unpool)
        self.D3 = DecoderBlock(self.L3_lap, in_c= 64+ 64, mid_c= 32, out_c= 16, unpooling=unpool)

        # Wrap up
        self.dec_fin_mu = torch.nn.Conv1d(16, 1, 1)
        self.dec_fin_logvar = torch.nn.Conv1d(16, 1, 1)
        self.dec_fin_logvar.weight.data.normal_(0, 1e-6)
        self.dec_fin_logvar.bias.data.fill_(0.01)
        self.cd_mu = SpatialConcreteDropout(channels_first=True)
        self.cd_logvar = SpatialConcreteDropout(channels_first=True)

    def forward(self, x):
        # Encoder
        x3 = self.E3(x)
        x2 = self.E2(x3)
        x1 = self.E1(x2)

        # Bottleneck
        x = self.B(x1)

        # Decoder
        x = self.D1(x, x1)
        x = self.D2(x, x2)
        x = self.D3(x, x3)

        x = x.permute(0, 2, 1)

        # Wrap up
        mu = self.cd_mu(lap=None, x=x, layer=self.dec_fin_mu)
        logvar = self.cd_logvar(lap=None, x=x, layer=self.dec_fin_logvar)

        return mu, logvar


model = DeepSphereUNetCD()
nside = 32  # hard-coded above
npix = hp.nside2npix(nside)
dummy_input = torch.randn(1, npix, NCH)  # 1 sample, NCH channels, npix pixels
# dummy_input = torch.randn(1, NCH, npix)  # 1 sample, NCH channels, npix pixels  # proposed

model(dummy_input)
