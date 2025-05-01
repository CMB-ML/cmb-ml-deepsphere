# From deepsphere implementation at https://github.com/deepsphere/deepsphere-pytorch
# 

"""
Chebyshev convolution layer. 
Taken from Michaël Defferrard's implementation, there has been minor adjustments to the cheb_conv fuction. 

Changes:
The cheb_conv function preallocates the Chebyshev basis to avoid memory spikes from torch.cat().
Added a safe_sparse_matmul function to ensure inputs and weights are cast to float32 before multiplication in use cases of mixed precision training.
SphericalChebConv class uses a LaplacianModule object to register the buffer instead of registering the buffer directly.
This avoids significant memory duplication when the model is moved to GPU.
"""
# pylint: disable=W0221

import math

import torch
from torch import nn

from .utils import LaplacianModule # For typing hint

def safe_sparse_matmul(A, B):
    device = A.device.type
    with torch.amp.autocast(device_type=device, enabled=False):
        if A.dtype == torch.float16 or B.dtype == torch.float16:
            A = A.float()
            B = B.float()    
        return torch.sparse.mm(A, B)

# Check memory efficiency of this function

def cheb_conv(laplacian, inputs, weight):
    """Chebyshev convolution.

    Args:
        laplacian (:obj:`torch.sparse.Tensor`): The laplacian corresponding to the current sampling of the sphere.
        inputs (:obj:`torch.Tensor`): The current input data being forwarded.
        weight (:obj:`torch.Tensor`): The weights of the current layer.

    Returns:
        :obj:`torch.Tensor`: Inputs after applying Chebyshev convolution.
    """
    B, V, Fin = inputs.shape
    K, Fin, Fout = weight.shape
    # B = batch size
    # V = nb vertices
    # Fin = nb input features
    # Fout = nb output features
    # K = order of Chebyshev polynomials

    # transform to Chebyshev basis
    x0 = inputs.permute(1, 2, 0)  # V x Fin x B
    x0 = x0.reshape([V, Fin * B])  # V x Fin*B
    inputs = x0.unsqueeze(0)  # 1 x V x Fin*B

    if K > 0:
        # x1 = torch.sparse.mm(laplacian, x0)  # V x Fin*B
        x1 = safe_sparse_matmul(laplacian, x0)  # V x Fin*B
        inputs = torch.cat((inputs, x1.unsqueeze(0)), 0)  # 2 x V x Fin*B
        for _ in range(1, K - 1):
            # x2 = 2 * torch.sparse.mm(laplacian, x1) - x0
            x2 = 2 * safe_sparse_matmul(laplacian, x1) - x0
            inputs = torch.cat((inputs, x2.unsqueeze(0)), 0)  # M x Fin*B
            x0, x1 = x1, x2
    
    del x0, x1, x2
    # final input shape is K x V x Fin*B

    inputs = inputs.view([K, V, Fin, B])  # K x V x Fin x B
    inputs = inputs.permute(3, 1, 2, 0)  # B x V x Fin x K
    inputs = inputs.reshape([B * V, Fin * K])  # B*V x Fin*K

    # Linearly compose Fin features to get Fout features
    weight = weight.view(Fin * K, Fout)
    inputs = inputs.matmul(weight)  # B*V x Fout
    inputs = inputs.view([B, V, Fout])  # B x V x Fout

    return inputs

def cheb_conv_prealloc(laplacian, inputs, weight):
    """Chebyshev convolution.

    Args:
        laplacian (:obj:`torch.sparse.Tensor`): The laplacian corresponding to the current sampling of the sphere.
        inputs (:obj:`torch.Tensor`): The current input data being forwarded.
        weight (:obj:`torch.Tensor`): The weights of the current layer.

    Returns:
        :obj:`torch.Tensor`: Inputs after applying Chebyshev convolution.
    """
    B, V, Fin = inputs.shape
    K, Fin, Fout = weight.shape
    # B = batch size
    # V = nb vertices
    # Fin = nb input features
    # Fout = nb output features
    # K = order of Chebyshev polynomials

    # transform to Chebyshev basis
    x0 = inputs.permute(1, 2, 0)  # V x Fin x B
    x0 = x0.reshape([V, Fin * B])  # V x Fin*B
    cheb_basis = torch.zeros(K, V, Fin * B, device=inputs.device)
    cheb_basis[0] = x0

    if K > 1:
        cheb_basis[1] = safe_sparse_matmul(laplacian, cheb_basis[0])  # V x Fin*B
        for i in range(2, K):
            cheb_basis[i] = 2 * safe_sparse_matmul(laplacian, cheb_basis[i-1]) - cheb_basis[i-2]
            
    inputs = cheb_basis
    inputs = inputs.view([K, V, Fin, B])  # K x V x Fin x B
    inputs = inputs.permute(3, 1, 2, 0)  # B x V x Fin x K
    inputs = inputs.reshape([B * V, Fin * K])  # B*V x Fin*K

    # Linearly compose Fin features to get Fout features
    weight = weight.view(Fin * K, Fout)
    inputs = inputs.matmul(weight)  # B*V x Fout
    inputs = inputs.view([B, V, Fout])  # B x V x Fout

    return inputs


class ChebConv(torch.nn.Module):
    """Graph convolutional layer.
    """

    def __init__(self, in_channels, out_channels, kernel_size, initialization='xnorm', bias=True, conv=cheb_conv_prealloc):
        """Initialize the Chebyshev layer.

        Args:
            in_channels (int): Number of channels/features in the input graph.
            out_channels (int): Number of channels/features in the output graph.
            kernel_size (int): Number of trainable parameters per filter, which is also the size of the convolutional kernel.
                                The order of the Chebyshev polynomials is kernel_size - 1.
            bias (bool): Whether to add a bias term.
            conv (callable): Function which will perform the actual convolution.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self._conv = conv

        shape = (kernel_size, in_channels, out_channels)
        self.weight = torch.nn.Parameter(torch.Tensor(*shape))

        if bias:
            self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter("bias", None)

        self.initialization = initialization
        if self.initialization == 'xunif':
            self.xavier_unif_initialization()
        elif self.initialization == 'xnorm':
            self.xavier_normal_initialization()
        elif self.initialization == 'kaiming':
            self.kaiming_initialization()

    def kaiming_initialization(self):
        """Initialize weights and bias.
        """
        std = math.sqrt(2 / (self.in_channels * self.kernel_size))
        self.weight.data.normal_(0, std)
        if self.bias is not None:
            self.bias.data.fill_(0.01)

    def xavier_unif_initialization(self):
        """Initialize weights and bias.
        """
        std = math.sqrt(6 / (self.in_channels + self.out_channels))
        self.weight.data.uniform_(-std, std)
        if self.bias is not None:
            self.bias.data.fill_(0.01)
    
    def xavier_normal_initialization(self):
        """Initialize weights and bias.
        """
        std = math.sqrt(2 / (self.in_channels + self.out_channels))
        self.weight.data.normal_(0, std)
        if self.bias is not None:
            self.bias.data.fill_(0.01)
    
    def forward(self, laplacian, inputs):
        """Forward graph convolution.

        Args:
            laplacian (:obj:`torch.sparse.Tensor`): The laplacian corresponding to the current sampling of the sphere.
            inputs (:obj:`torch.Tensor`): The current input data being forwarded.

        Returns:
            :obj:`torch.Tensor`: The convoluted inputs.
        """
        outputs = self._conv(laplacian, inputs, self.weight)
        if self.bias is not None:
            outputs += self.bias
        return outputs

class SphericalChebConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, laplacian: LaplacianModule, bias: bool = True):
        """Initialize the Spherical Chebyshev layer.

        Args:
            in_channels (int): Number of channels/features in the input graph.
            out_channels (int): Number of channels/features in the output graph.
            kernel_size (int): Number of trainable parameters per filter, which is also the size of the convolutional kernel.
                                The order of the Chebyshev polynomials is kernel_size - 1.
            laplacian (LaplacianModule): Laplacian matrix for the spherical graph.
            bias (bool): Whether to add a bias term.
        """
        super().__init__()
        self.laplacian = laplacian
        self.cheb_conv = ChebConv(in_channels, out_channels, kernel_size, bias=bias)

    def forward(self, x):
        """Forward graph convolution.

        Args:
            x (:obj:`torch.Tensor`): The current input data being forwarded.

        Returns:
            :obj:`torch.Tensor`: The convoluted inputs.
        """
        laplacian = getattr(self.laplacian, f"laplacian_{self.laplacian.laplacian_idx}")
        return self.cheb_conv(laplacian, x)