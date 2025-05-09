# From deepsphere implementation at https://github.com/deepsphere/deepsphere-pytorch
# 

import torch
from torch import nn

class LaplacianModule(nn.Module):
    """
    Wrapper class to handle storing Laplacian matrices without duplication
    """
    def __init__(self, laplacian: torch.Tensor, laplacian_idx: int):
        """
        Args:
            laplacian (torch.Tensor): Laplacian matrix.
            laplacian_idx (int): Index of the Laplacian matrix for unique name.
        """
        super().__init__()
        self.laplacian_idx = laplacian_idx
        self.register_buffer(f"laplacian_{laplacian_idx}", laplacian)
class HealpixBatchNorm(nn.BatchNorm1d):
    def forward(self, x):
        """Forward pass of the batch normalization layer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch x vertices x channels].

        Returns:
            torch.Tensor: Normalized tensor.
        """
        # Permute the input tensor to match the expected shape for BatchNorm1d
        x = x.permute(0, 2, 1)
        # Apply batch normalization
        x = super().forward(x)
        # Permute back to original shape
        return x.permute(0, 2, 1)

class HealpixDownsample(nn.MaxPool1d):
    def __init__(self, kernel_size=4, return_indices=False):
        """Initialization of the max pooling layer.

        Args:
            kernel_size (int, optional): Size of the pooling kernel. Defaults to 4.
            return_indices (bool, optional): Whether to return indices. Defaults to False.
        """
        super().__init__(kernel_size=kernel_size, return_indices=return_indices)

    def forward(self, x):
        """Forward pass of the max pooling layer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch x vertices x channels].

        Returns:
            torch.Tensor: Pooled tensor.
        """
        # Permute the input tensor to match the expected shape for MaxPool1d
        x = x.permute(0, 2, 1)
        # Apply max pooling
        x = super().forward(x)
        # Permute back to original shape
        return x.permute(0, 2, 1)

class HealpixUpsample(nn.Upsample):
    def __init__(self, scale_factor=4, mode='nearest'):
        """Initialization of the upsampling layer.

        Args:
            scale_factor (int, optional): Scale factor for upsampling. Defaults to None.
            mode (str, optional): Upsampling mode. Defaults to 'nearest'.
        """
        super().__init__(scale_factor=scale_factor, mode=mode)

    def forward(self, x):
        """Forward pass of the upsampling layer.

        Args:
            x (torch.Tensor): Input tensor of shape [batch x vertices x channels].

        Returns:
            torch.Tensor: Upsampled tensor.
        """
        # Permute the input tensor to match the expected shape for Upsample
        x = x.permute(0, 2, 1)
        # Apply upsampling
        x = super().forward(x)
        # Permute back to original shape
        return x.permute(0, 2, 1)
    
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