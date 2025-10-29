"""
Contains components of the ConvNeXt model architecture
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath

class Block(nn.Module):
    """
    ConvNeXt Block. Implemented by:
    DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back

    Args:
        dim (int): Number of input channels
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        """
        Forward pass for the ConvNeXt block

        Args:
            x (Tensor): Input tensor of shape (N, C, H, W)
        
        Returns:
            Tensor: output tensor of shape (N, C, H, W)
        """
        input = x
        x = self.dwconv(x)

        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x

        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x 
    
class ConvNeXt(nn.Module):
    """ 
    ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf

    Args:
        in_chans (int): Number of input image channels. Default = 3
        num_classes (int): Number of classes for classification head. Default: 2
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        """
        Initialises weights using truncated normal distributions and biases to zero for convolutional and linear layers
        """
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        """
        Propagates input through the model's feature extraction stages and applies Global Average Pooling

        Args:
            x (Tensor): Input tensor of shape (N, C, H, W)

        Returns:
            Tensor: Feature vector of shape (N,C) after GAP and final norm
        """
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        """
        Full forward pass for classification

        Args:
            x (Tensor): Input tensor of shape (N, C, H, W)

        Returns:
            Tensor: output (logits) of shape (N, num_classes)
        """
        x = self.forward_features(x)
        x = self.head(x)
        return x

class LayerNorm(nn.Module):
    """ 
    LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).

    ArgsL
        normalised_shape (int): the number of features (channels) to normalise
        eps (float): a value added to the denominator for numerical stablilty. Default=1e-6
        data_format (str): The input dataformat, either "channels_last" or "channels_first". Default="channels_last"
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        """
        Applies Layer Normalisation based on the specified data format

        Args:
            x (Tensor): Input tensor

        Returns:
            Tensor: Normalised output tensor
        """
        if self.data_format == "channels_last":
            # uses pytorch's layerNorm for channels_last (normalisation over the last dimension)
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        
        elif self.data_format == "channels_first":
            # Manual calculation for normalisation over the channel dimension C=1
            u = x.mean(1, keepdim=True)     # calculate mean across the channel dimension
            s = (x - u).pow(2).mean(1, keepdim=True)    # calculate variance 
            x = (x - u) / torch.sqrt(s + self.eps)  # normalise
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def convnext_tiny():
    """
    Creates a ConvNeXt-Tiny model instance.

    Args:
        in_chans (int): Number of input channels. Default: 1.
        num_classes (int): Number of output classes. Default: 2.
    
    Returns:
        ConvNeXt: The initialized ConvNeXt-Tiny model.
    """
    return ConvNeXt(in_chans=1, num_classes=2)

def covnext_small(drop_path_rate):
    """
    Creates a ConvNeXt-Small model instance.
    
    This configuration uses depths=[3, 3, 27, 3] and dims=[96, 192, 384, 768].
    It is configured for 1 input channel (grayscale) and 2 output classes (AD/NC) to match the ADNI dataset context.

    Args:
        drop_path_rate (float): Stochastic depth rate.
        
    Returns:
        ConvNeXt: The initialized ConvNeXt-Small model.
    """
    return ConvNeXt(in_chans=1, num_classes=2, depths = [3, 3, 27, 3], dims = [96, 192, 384, 768], drop_path_rate=drop_path_rate, layer_scale_init_value=1e-6, head_init_scale=1.)
