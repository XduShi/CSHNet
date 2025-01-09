import torch
from torch._C import TracingState
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from torch.nn import modules
from torch.nn.modules import padding

#from FGGAN.sota.CycleGAN.models.networks import PixelDiscriminator
from .layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
import torch.nn.functional as F
import math
from focal_frequency_loss import FocalFrequencyLoss as FFL

###############################################################################
# Functions
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer
###定义Transformer主要部分

##定义PatchEmbed，用于将图像进行分块和嵌入
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        input_nc (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, img_size=128, patch_size=4, input_nc=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.input_nc = input_nc
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(input_nc, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)  # B Ph*Pw C
        if self.norm is not None:
            x = self.norm(x)
        return x  #(B,3,256,256)->(B,96,64,64)->(B,96,64*64)->(B,64*64,96)


##定义PatchUnEmbed  用于卷积操作
class PatchUnEmbed(nn.Module):
    r""" Image to Patch Unembedding
    Args:
        img_size (int): Image size.  Default: 224.
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, input_resolution,img_size=128, patch_size=4, input_nc=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.input_nc = input_nc
        self.embed_dim = embed_dim
        self.input_resolution = input_resolution
        

    def forward(self, x):
        B, HW, C = x.shape
        H, W = self.input_resolution
        x = x.transpose(1, 2).view(B, self.embed_dim, self.input_resolution[0], self.input_resolution[1])  
        return x


##定义MLPConv 在两层线性层之间加入卷积，增加局部上下文联系
class MlpC(nn.Module):  # Mlp inside encoder
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.conv = nn.Conv2d(hidden_features, hidden_features, kernel_size=3,stride=1,padding=1, groups= hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        B, HW, C = x.size()
        HH = int(math.sqrt(HW))
        x = rearrange(x, ' B (H W) (C) -> B C H W ', H = HH, W = HH)
        x = self.conv(x)
        x = self.act(x)
        x = self.drop(x)
        # flaten
        x = rearrange(x, ' B C H W -> B (H W) (C)', H = HH, W = HH)
        x = self.fc2(x)
        x = self.drop(x)
        return x
##定义MLP
class Mlp(nn.Module):  # Mlp inside encoder
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
##分割窗口，用于计算局部attention
def window_partition(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

##恢复窗口
def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        H (int): Height of image
        W (int): Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x

##定义attention
class WindowAttention(nn.Module):
    r""" Window based multi-head self attention (W-MSA) module with relative position bias.
    It supports both of shifted and non-shifted window.
    Args:
        dim (int): Number of input channels.
        window_size (tuple[int]): The height and width of the window.
        num_heads (int): Number of attention heads.
        qkv_bias (bool, optional):  If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set
        attn_drop (float, optional): Dropout ratio of attention weight. Default: 0.0
        proj_drop (float, optional): Dropout ratio of output. Default: 0.0
    """

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5 #^(-1/2)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 将（Wh,Wh) 与 (Wh,Wh)在dim = 0维度合并->2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww 利用广播机制，分别在第一维，第二维，插入一个维度，进行广播相减
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index) #不参与网络学习的变量

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

###PatchMerging 用于合并Patch 相当于下采样
class PatchMerging(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        assert H % 2 == 0 and W % 2 == 0, f"x size ({H}*{W}) are not even."

        x = x.view(B, H, W, C)

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)  # (B, H/4, W/4, C)->(B, H/8, W/8, 2C)

        return x

###定义下采样DownSampling 替换PatchMerging edit-2021-10-19

class DownSampling(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.conv1=nn.Conv2d(dim,2*dim,kernel_size=3,stride=2,padding=1)
        #self.norm = nn.BatchNorm2d(2 * dim)
        self.norm = norm_layer(2 * dim)
        #self.relu=nn.ReLU()
    def forward(self,x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C).permute(0,3,1,2)
        x=self.conv1(x)        
        
        x = x.permute(0,2,3,1).view(B, -1, 2 * C)
        x=self.norm(x)
        #x=self.relu(x)
        return x

###PatchExpand用于上采样，扩大patch尺寸
class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2*dim, bias=False) if dim_scale==2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C//4)
        x = x.view(B,-1,C//4)
        x= self.norm(x)

        return x

#定义上采样UpSampling，替换PatchExpand edit-2021-10-19
class UpSampling(nn.Module):
    def __init__(self, input_resolution, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.convT = nn.ConvTranspose2d(dim, int(dim / 2), kernel_size=3, stride=2, padding=1, output_padding=1)
        self.norm = norm_layer(int(dim/2))
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C).permute(0,3,1,2)
        x = self.convT(x)
        x = x.permute(0,2,3,1).view(B, -1,  C//2)
        x = self.norm(x)
        
        return x


###patch_size = 4 时dim_scale=4， nn.Linear(dim, 16*dim, bias=False)
###patch_size = 2 时dim_scale=2， nn.Linear(dim, 4*dim, bias=False)
class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.dim_scale = dim_scale
        self.expand = nn.Linear(dim, 16*dim, bias=False)
        self.output_dim = dim 
        self.norm = norm_layer(self.output_dim)
        self.tanh = nn.Tanh()

    def forward(self, x):
        """
        x: B, H*W, C
        """
        H, W = self.input_resolution
        x = self.expand(x)
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale, c=C//(self.dim_scale**2))
        x = x.view(B,-1,self.output_dim)
        x = self.norm(x)
        x = self.tanh(x)

        return x

#定义Final_up 最后一层上采样 替换 FinalPatchExpand_X4 edit-2021-10-19
class Final_up(nn.Module):
    def __init__(self, input_resolution, dim):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.final_conv = nn.ConvTranspose2d(dim, 64, kernel_size=7, stride=4, padding=2, output_padding=1)
    def forward(self, x):
        H, W = self.input_resolution
        B, L, C = x.shape
        x = x.view(B, H, W, C).permute(0,3,1,2)
        x = self.final_conv(x)
        return x

###Swin-Transformer Block
class SwinTransformerBlock(nn.Module):
    r""" Swin Transformer Block.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resulotion.
        num_heads (int): Number of attention heads.
        window_size (int): Window size.
        shift_size (int): Shift size for SW-MSA.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float, optional): Stochastic depth rate. Default: 0.0
        act_layer (nn.Module, optional): Activation layer. Default: nn.GELU
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, input_resolution, num_heads, window_size=8, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size: #--edit2021/11/2
        #if self.input_resolution <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MlpC(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution  #--edit2021/11/2
            #H = self.input_resolution
            #W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

    
        

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):
        #edit 2021/11/2
        H, W = self.input_resolution
        #H = self.input_resolution
        #W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x
###定义下采样层
class BasicLayer(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])
        #############################在每个SwinTB之后添加一个卷积，用来加强特征(不太行）############
        #self.conv = nn.Conv2d(dim, dim, 3, 1, 1)

        

        #self.patch_unembed = PatchUnEmbed(input_resolution=input_resolution,
        #    img_size=img_size, patch_size=patch_size, input_nc=dim, embed_dim=dim,
        #    norm_layer=None)
        
        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        ####x输入卷积层#######################
        #x=self.patch_unembed(x)
        #x=self.conv(x)
        #H, W = self.input_resolution
        #B=1
        #C=x.size()[1]
        #x = x.view(B,H*W,C)
        
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        

        if self.downsample is not None:
            x = self.downsample(x)
        return x

#上采样层
class BasicLayer_up(nn.Module):
    """ A basic Swin Transformer layer for one stage.
    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, window_size, img_size, patch_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, upsample=None, use_checkpoint=False):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias, qk_scale=qk_scale,
                                 drop=drop, attn_drop=attn_drop,
                                 drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                                 norm_layer=norm_layer)
            for i in range(depth)])
        #############################在每个SwinTB之后添加一个卷积，用来加强特征############
        #self.conv = nn.Conv2d(dim, dim, 3, 1, 1)
        
        #self.patch_unembed = PatchUnEmbed(input_resolution=input_resolution,
        #    img_size=img_size, patch_size=patch_size, input_nc=dim, embed_dim=dim,
        #    norm_layer=None)

        # patch merging layer
        if upsample is not None:
            self.upsample = UpSampling(input_resolution, dim=dim,  norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):

         ####x输入卷积层#######################
        #x=self.patch_unembed(x)
        #x=self.conv(x)
        #H, W = self.input_resolution
        #B=1
        #C=x.size()[1]
        #x = x.view(B,H*W,C)
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
       

        if self.upsample is not None:
            x = self.upsample(x)
        return x

# define_G函数里面主要内容就是这几行，我们知道在Pix2PixHD中，G是有两部分的，
# 一部分是global net，
# 另一部分是local net，就是前两个if语句对应的分支。
# 第三个if语句对应的是论文中E的部分，用来预先计算类别特征，实现同样的semantic label的多样性输出。
#############################################################################
##原码定义生成器
#def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1, 
#             n_blocks_local=3, norm='instance', gpu_ids=[]):    
#    norm_layer = get_norm_layer(norm_type=norm)     
#    if netG == 'global':    
#        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)       
#    elif netG == 'local':        
#        netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, 
#                                  n_local_enhancers, n_blocks_local, norm_layer)
#    elif netG == 'encoder':
#        netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
#    else:
#        raise('generator not implemented!')
#    print(netG)
#    if len(gpu_ids) > 0:
#        assert(torch.cuda.is_available())   
#        netG.cuda(gpu_ids[0])
#    netG.apply(weights_init)
#    return netG

###############################################################################
##定义新的生成器，采用Transformer架构 
def define_G(ngf, input_nc, output_nc, n_downsampling,n_blocks,
                 window_size, mlp_ratio=4., 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                  ape=False, patch_norm=True,
                 use_checkpoint=False,qkv_bias=True, qk_scale=None):
    
    #netG = TransG(img_size=256, patch_size=4, input_nc=3, output_nc=3,
    #             embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
    #             window_size=16, mlp_ratio=4., 
    #             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
    #             norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
    #             use_checkpoint=False, final_upsample="expand_first",qkv_bias=True, qk_scale=None,)
    #netG = TransGHD(ngf=32, n_local_enhancers=1, n_block_local=3,norm_layer=nn.InstanceNorm2d,padding_type='reflect',
    #             img_size=128, patch_size=4, input_nc=3, output_nc=3,
    #             embed_dim=96, depths=[2, 2, 2, 2],depths_decoder=[1, 2, 2, 2],num_heads=[3, 6, 12, 24],
    #             window_size=16, mlp_ratio=4., 
    #             drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
    #              ape=False, patch_norm=True,
    #             use_checkpoint=False, final_upsample="expand_first",qkv_bias=True, qk_scale=None)
    #netG = GlobalT(input_nc, output_nc, ngf, n_downsampling=3, n_blocks=4, norm_layer=nn.BatchNorm2d, 
    #             num_heads=8, window_size=8,padding_type='reflect')
    ####采用ResTrans模型--edit2021/11/10
    netG = ResTransHD(input_nc, output_nc, ngf, n_downsampling=2, n_blocks=4, num_heads=16, window_size=16)
    print(netG)
    #if len(gpu_ids) > 0:
    #    assert(torch.cuda.is_available())   
    #    netG.cuda(gpu_ids[0])
    netG.apply(weights_init)
    return netG
############################################################################
##判别器的结构先不改变
def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False, gpu_ids=[]):        
    norm_layer = get_norm_layer(norm_type=norm)   
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)   
    #print(netD)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        netD.cuda(gpu_ids[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

##############################################################################
# Losses
##############################################################################
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1, target_fake_label=0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:  # Ture
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]
                target_tensor = self.get_target_tensor(pred, target_is_real)
                loss += self.loss(pred, target_tensor)
            return loss
        else:            
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()        
        self.vgg = Vgg19().cuda()
        self.criterion = nn.L1Loss()
        self.weights = [1.0/32, 1.0/16, 1.0/8, 1.0/4, 1.0]        

    def forward(self, x, y):              
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())        
        return loss
#############################################################################
##############################################################################
# Generator
##############################################################################

###########定义含有Swin-Transformer的生成器#######
class TransG(nn.Module):
    def __init__(self, img_size=128, patch_size=4, input_nc=3, output_nc=3,
                 embed_dim=96, depths=[2, 2, 2, 2], depths_decoder=[1, 2, 2, 2], num_heads=[3, 6, 12, 24],
                 window_size=16, mlp_ratio=4., 
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 use_checkpoint=False, final_upsample="expand_first", qkv_bias=True, qk_scale=None,**kwargs):
        super().__init__()

        print("TransG expand initial----depths:{};depths_decoder:{};drop_path_rate:{};output_nc:{}".format(depths,
        depths_decoder,drop_path_rate,output_nc))

        self.output_nc = output_nc
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.num_features_up = int(embed_dim * 2)
        self.mlp_ratio = mlp_ratio
        self.final_upsample = final_upsample

        # 对图像进行分块嵌入 split image into non-overlapping patches
        
        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, input_nc=input_nc, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        # 绝对位置编码absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # 建立编码器build encoder 
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dim * 2 ** i_layer),
                               input_resolution=(patches_resolution[0] // (2 ** i_layer),
                                                 patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=DownSampling if (i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
            self.layers.append(layer)
        
        # 创建解码器build decoder layers
        self.layers_up = nn.ModuleList()
        self.concat_back_dim = nn.ModuleList()
        for i_layer in range(self.num_layers):
            concat_linear = nn.Linear(2*int(embed_dim*2**(self.num_layers-1-i_layer)),
            int(embed_dim*2**(self.num_layers-1-i_layer))) if i_layer > 0 else nn.Identity()
            if i_layer ==0 :
                layer_up = UpSampling(input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))), dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)),  norm_layer=norm_layer)
            else:
                layer_up = BasicLayer_up(dim=int(embed_dim * 2 ** (self.num_layers-1-i_layer)),
                                input_resolution=(patches_resolution[0] // (2 ** (self.num_layers-1-i_layer)),
                                                    patches_resolution[1] // (2 ** (self.num_layers-1-i_layer))),
                                depth=depths[(self.num_layers-1-i_layer)],
                                num_heads=num_heads[(self.num_layers-1-i_layer)],
                                window_size=window_size,
                                img_size=img_size,
                                patch_size=patch_size,
                                mlp_ratio=self.mlp_ratio,
                                qkv_bias=qkv_bias, qk_scale=qk_scale,
                                drop=drop_rate, attn_drop=attn_drop_rate,
                                drop_path=dpr[sum(depths[:(self.num_layers-1-i_layer)]):sum(depths[:(self.num_layers-1-i_layer) + 1])],
                                norm_layer=norm_layer,
                                upsample=UpSampling if (i_layer < self.num_layers - 1) else None,
                                use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)
            self.concat_back_dim.append(concat_linear)

        self.norm = norm_layer(self.num_features)
        self.norm_up= norm_layer(self.embed_dim)

        if self.final_upsample == "expand_first":
            print("---final upsample expand_first---")
            #self.up = FinalPatchExpand_X4(input_resolution=(img_size//patch_size,img_size//patch_size),dim_scale=4,dim=embed_dim)
            #self.output = nn.Conv2d(in_channels=embed_dim,out_channels=self.output_nc,kernel_size=1,bias=False)
            ##采用Final_up
            self.output = Final_up(input_resolution=(img_size//patch_size,img_size//patch_size), dim=embed_dim)
            self.tanh = nn.Tanh()

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    #Encoder and Bottleneck
    def forward_features(self, x):
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        x_downsample = []

        for layer in self.layers:
            x_downsample.append(x)
            x = layer(x)

        x = self.norm(x)  # B L C
  
        return x, x_downsample

    #Dencoder and Skip connection
    def forward_up_features(self, x, x_downsample):
        for inx, layer_up in enumerate(self.layers_up):
            if inx == 0:
                x = layer_up(x)
            else:
                x = torch.cat([x,x_downsample[3-inx]],-1)
                x = self.concat_back_dim[inx](x)
                x = layer_up(x)

        x = self.norm_up(x)  # B L C
  
        return x

    def up_x4(self, x):
        H, W = self.patches_resolution
        B, L, C = x.shape
        assert L == H*W, "input features has wrong size"

        if self.final_upsample=="expand_first":

            x = self.output(x)
            x = self.tanh(x)
            #x = x.view(B,4*H,4*W,-1) ##patch_size=4
            #x = x.view(B,2*H,2*W,-1) ##patch_size=2
            #x = x.permute(0,3,1,2) #B,C,H,W
           # x = self.output(x)
            
        return x

    def forward(self, x):
        x, x_downsample = self.forward_features(x)
        x = self.forward_up_features(x, x_downsample)
        x = self.up_x4(x)

        return x
    

##运用p2pHD的思想----edit 2021/10/28
class TransGHD(nn.Module):
    def __init__(self, input_nc=3,output_nc =3,ngf=32, n_local_enhancers=1, n_block_local=3,norm_layer=nn.BatchNorm2d,padding_type='reflect',**kwargs):
        super().__init__()
        self.n_local_enhancers=n_local_enhancers
        ##全局关注采用TransG
        model_g = TransG( input_nc=3, output_nc=3,  norm_layer=nn.LayerNorm)
        self.model_G = model_g
        ##local enhancer layers
        for n in range(1, n_local_enhancers+1):
            ##downsampling
            ngf_g = ngf * (2**(n_local_enhancers-n))
            model_ds = [nn.ReflectionPad2d(3),nn.Conv2d(input_nc, ngf_g, kernel_size=7,padding=0),
                        norm_layer(ngf_g), nn.ReLU(True),
                        nn.Conv2d(ngf_g, ngf_g*2,kernel_size=3,stride=2,padding=1),
                        norm_layer(ngf_g*2),nn.ReLU(True)]
            ##upsampling+residual block
            model_us=[]
            ##residual block
            for i in range(n_block_local):
                model_us += [ResnetBlock(ngf_g*2, padding_type=padding_type, norm_layer=norm_layer)]
            ##upsampling
            model_us += [nn.ConvTranspose2d(ngf_g*2, ngf_g, kernel_size=3, stride=2,padding=1,output_padding=1),
                         norm_layer(ngf_g),nn.ReLU(True)]
            if n == n_local_enhancers:
                model_us += [nn.ReflectionPad2d(3), nn.Conv2d(ngf_g, output_nc, kernel_size=7,padding=0),nn.Tanh()]
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_ds)) #setattr(object, name, value):该函数给对象中的属性赋值，该属性若不存在，则会在对象里创建新的属性 #创建self.model1_1=nn.Sequential(*model_ds)
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_us)) #创建self.model1_2=nn.Sequential(*model_us)
        ##将输入下采样后输入TransG
        self.downsampling=nn.AvgPool2d(3,stride=2,padding=[1,1],count_include_pad=False)
        ##将TransG输出与model_ds输出concat后降维
        self.concat=nn.Conv2d(ngf_g*4, ngf_g*2,kernel_size=3,stride=1,padding=1)
    
    def forward(self, input):
        input_ds=[input]
        for i in range(self.n_local_enhancers):
            input_ds.append(self.downsampling(input_ds[-1])) #对输入进行2倍下采样后输入TransG
        output_G= self.model_G(input_ds[-1])
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_L_ds = getattr(self,'model'+str(n_local_enhancers)+'_1')  #getattr(object, name, [default]) object为对象名，name为对象属性（必须是字符串）若对象中存在name属性，返回object.name ##返回model1_1
            model_L_us = getattr(self,'model'+str(n_local_enhancers)+'_2')  #返回model1_2
            input_i = input_ds[self.n_local_enhancers-n_local_enhancers]
            middle_L = model_L_ds(input_i)
            cat = torch.cat([middle_L, output_G], 1)
            out_cat = self.concat(cat)
            output = model_L_us(out_cat)
        return output

       

##探索Transformer和residual block的关系 将residual block 替换为swin block 利用GlobalG架构--edit2021/11/2
class GlobalT(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=4, norm_layer=nn.BatchNorm2d, 
                  num_heads=8, window_size=8,padding_type='reflect'):
        assert(n_blocks >= 0)
        super().__init__()        
        activation = nn.ReLU(True)        
        self.model1 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation)
        ### downsample
        
        model_d=[]
        for i in range(n_downsampling):
            mult = 2**i
            model_d += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]
        self.model_D = nn.Sequential(*model_d)
        mult = 2**n_downsampling
        model_mr1=[]
        model_ms=[]
        model_mr2=[]
        ##residual block
        for i in range(n_blocks):
            model_mr1 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer)]
        ###swinBlock
        #for i in range(n_blocks):
        #    model_ms += [SwinTransformerBlock(dim, input_resolution, num_heads, window_size, shift_size=0 if (i % 2 == 0) else window_size // 2,
        #         mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
        #         act_layer=nn.GELU, norm_layer=nn.LayerNorm)]
        for i in range(2):
            model_ms += [SwinTransformerBlock(dim=ngf*8, input_resolution=(32,32),
                                 num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=4.0,
                                 qkv_bias=True, qk_scale=None,
                                 drop=0.0, attn_drop=0.0,
                                 drop_path=0.2,
                                 norm_layer=nn.LayerNorm)]
        ##residual block
        for i in range(n_blocks):
            model_mr2 += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer)]
        self.model_MR1 = nn.Sequential(*model_mr1)
        self.model_MS = nn.Sequential(*model_ms)
        self.model_MR2 = nn.Sequential(*model_mr2)
        
        ### upsample 
        model_u=[]        
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model_u += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        self.model_U = nn.Sequential(*model_u)
        self.model2 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh())       
        #self.model = nn.Sequential(*model)
            
    def forward(self, x):
        x = self.model1(x)
        x = self.model_D(x)
        x = self.model_MR1(x)
        B,C,H,W = x.shape
        x = x.permute(0,2,3,1).view(B, -1, C)
        x = self.model_MS(x)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)
        x = self.model_MR2(x)
        x = self.model_U(x)
        x = self.model2(x)


        return x 


################-结合Transformer的长期依赖和CNN的局部细节(卷积和SwinBlock结合)-2021/11/10######################
##定义conv和Transformer模块
class ResSA(nn.Module):
    def __init__(self, dim, activation = nn.ReLU(True), num_heads=16, window_size=16):
        super().__init__()
        ###定义类似于residual block res_block
        norm_layer1 = nn.BatchNorm2d
        res_block = []
        res_block += [nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1), norm_layer1(dim),activation]
        res_block += [nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1), norm_layer1(dim)]
        self.Res_Block = nn.Sequential(*res_block)
        ###定义swin block
        swin_block = []
        ngf = 64
        for i in range(2):
            swin_block +=[SwinTransformerBlock(dim=ngf*4, input_resolution=(64,64),
                         num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=4.0,
                                 qkv_bias=True, qk_scale=None,
                                 drop=0.0, attn_drop=0.0,
                                 drop_path=0.2,
                                 norm_layer=nn.LayerNorm)]
        self.Swin_Block = nn.Sequential(*swin_block)
        ###3*3conv降维
        self.conv = nn.Conv2d(2*dim, dim, kernel_size=3, stride=1, padding=1)
    
    def forward(self,x):
        ##分支1经过卷积提取局部细节信息
        x1 = self.Res_Block(x)
        ##分支2经过Transformer建立长程依赖关系
        B, C, H, W = x.shape
        x2 = x.permute(0,2,3,1).view(B,-1,C)
        x2 = self.Swin_Block(x2)
        x2 = x2.view(B, H, W, C).permute(0,3,1,2)
        ##两个分支合并
        x3 = torch.cat([x1,x2],1)
        ##降维
        x4 = self.conv(x3)
        out = x + x4
        return out 
##定义包含ResSAR模块的生成器
class ResTrans(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=2, n_blocks=4, num_heads=16, window_size=16):
        super().__init__()
        activation = nn.ReLU(True)
        norm_layer1 = nn.BatchNorm2d
        ##1*64*256*256
        self.model1 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer1(ngf), activation)
        ##down_sampling
        model_d = []
        for i in range(n_downsampling):
            mult = 2**i
            model_d += [nn.Conv2d(ngf*mult, ngf*mult*2, kernel_size=3,stride=2, padding=1), norm_layer1(mult*ngf*2), activation]
        self.model_D = nn.Sequential(*model_d)
        ##中间过渡层，采用ResSAR模块
        model_m = []
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model_m += [ResSA(mult*ngf, activation=activation, num_heads=num_heads, window_size=window_size)]
        self.model_M = nn.Sequential(*model_m)
        ##up_sampling
        model_u = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling-i)
            model_u += [nn.ConvTranspose2d(ngf*mult, int(ngf*mult/2), kernel_size=3, stride=2, padding=1, output_padding=1),
                        norm_layer1(int(ngf*mult/2)), activation]
        self.model_U = nn.Sequential(*model_u)
        self.model2 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh())
    def forward(self,x):
        x = self.model1(x)
        x = self.model_D(x)
        x = self.model_M(x)
        x = self.model_U(x)
        x = self.model2(x)
        return x

##########为了进一步加强生成图像的细节，考虑在ResTrans中添加Encoder和Decoder之间的skip-connection#####---edit 2021/11/15---不太行失败
##定义下采样--2021/11/15
class DS(nn.Module):
    def __init__(self, in_dim, out_dim, norm_layer, activation):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm_layer = norm_layer
        self.activation = activation
        self.down = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)
    def forward(self,x):
        x = self.down(x)
        x = self.norm_layer(x)
        x = self.activation(x)
        return x
##定义decoder前的卷积模块Hold--2021/11/15
class Hold(nn.Module):
    def __init__(self, in_dim, out_dim, norm_layer, activation):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm_layer = norm_layer
        self.activation = activation
        self.down = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=1, padding=1)
    def forward(self,x):
        x = self.down(x)
        x = self.norm_layer(x)
        x = self.activation(x)
        return x
##定义上采样--2021/11/15
class US(nn.Module):
    def __init__(self, in_dim, out_dim, norm_layer, activation):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm_layer = norm_layer
        self.activation = activation
        self.up = nn.ConvTranspose2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1, output_padding=1)
    def forward(self,x):
        x = self.up(x)
        x = self.norm_layer(x)
        x = self.activation(x)
        return x
class ResTransHD(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=2, n_blocks=4, num_heads=16, window_size=16):
        super().__init__()
        activation = nn.ReLU(True)
        norm_layer1 = nn.BatchNorm2d
        ##1*64*256*256
        self.model1 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer1(ngf), activation)
        ##down_sampling
        model_d = []
        for i in range(n_downsampling):
            mult = 2**i
            model_d += [DS(ngf*mult, ngf*mult*2, norm_layer1(mult*ngf*2), activation)]
        self.model_D = nn.Sequential(*model_d)
        ##中间过渡层，采用ResSAR模块
        model_m = []
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model_m += [ResSA(mult*ngf, activation=activation, num_heads=num_heads, window_size=window_size)]
        self.model_M = nn.Sequential(*model_m)
        ##up_sampling
        model_u = []
        ##为了让Encoder和Decoder的层数匹配 在Decoder中添加一层64*64*256 --2021/11/15
        model_u += [Hold(mult*ngf, mult*ngf,norm_layer1(int(mult*ngf)), activation)]
        for i in range(n_downsampling):
            mult = 2**(n_downsampling-i)
            model_u += [US(ngf*mult, int(ngf*mult/2), norm_layer1(int(ngf*mult/2)), activation)]
        self.model_U = nn.Sequential(*model_u)
        self.model2 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh())
        ###为了concat降维
        self.concat_dim = nn.ModuleList()
        for i in range(n_downsampling+1):  ##0 1 2 
            dim = ngf *2**(n_downsampling+1-i)
            concat = nn.Conv2d(dim, int(dim/2), kernel_size=3, stride=1, padding=1)
            self.concat_dim.append(concat)
    ###定义Encoder和用于从concat的特征图
    def forward_features(self,x):
        x_downsample = []
        x = self.model1(x)
        x_downsample.append(x)
        for layer in self.model_D:
            x  = layer(x)
            x_downsample.append(x)
        return x, x_downsample
    ###定义采用concat特征图的Decoder
    def forward_up(self, x, x_downsample):
        for inx, layer_up in enumerate(self.model_U): 
            ##只concat两层(128*128*128和64*64*256)
            ##只concat一层（64*64*256）2021/11/21
            if inx >= 1:
                x = layer_up(x)
            else:  
                x = layer_up(x)
                x = torch.cat([x, x_downsample[2-inx]], 1)
                x = self.concat_dim[inx](x)
        return x
            
    ###定义整体的forward
    def forward(self, x):
        x, x_downsample = self.forward_features(x)
        x = self.model_M(x)
        x = self.forward_up(x, x_downsample)
        x = self.model2(x)
        return x
            

class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9, 
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=nn.BatchNorm2d, padding_type='reflect'):        
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers
        
        ###### global generator model #####           
        ngf_global = ngf * (2**n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global, norm_layer).model        
        model_global = [model_global[i] for i in range(len(model_global)-3)] # get rid of final convolution layers        
        self.model = nn.Sequential(*model_global)                

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers+1):
            ### downsample            
            ngf_global = ngf * (2**(n_local_enhancers-n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0), 
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1), 
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1), 
                               norm_layer(ngf_global), nn.ReLU(True)]      

            ### final convolution
            if n == n_local_enhancers:                
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]                       
            
            setattr(self, 'model'+str(n)+'_1', nn.Sequential(*model_downsample))
            setattr(self, 'model'+str(n)+'_2', nn.Sequential(*model_upsample))                  
        
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def forward(self, input): 
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])        
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers+1):
            model_downsample = getattr(self, 'model'+str(n_local_enhancers)+'_1')
            model_upsample = getattr(self, 'model'+str(n_local_enhancers)+'_2')            
            input_i = input_downsampled[self.n_local_enhancers-n_local_enhancers]            
            output_prev = model_upsample(model_downsample(input_i) + output_prev)
        return output_prev

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=nn.BatchNorm2d, 
                 padding_type='reflect'):
        assert(n_blocks >= 0)
        super(GlobalGenerator, self).__init__()        
        activation = nn.ReLU(True)        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), activation]

        ### resnet blocks
        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, activation=activation, norm_layer=norm_layer)]
        
        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), activation]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]        
        self.model = nn.Sequential(*model)
            
    def forward(self, input):
        return self.model(input)             
        
# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out

class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=nn.BatchNorm2d):
        super(Encoder, self).__init__()        
        self.output_nc = output_nc        

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), 
                 norm_layer(ngf), nn.ReLU(True)]             
        ### downsample
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample         
        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1),
                       norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]        

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model) 

    def forward(self, input, inst):
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()
        inst_list = np.unique(inst.cpu().numpy().astype(int))        
        for i in inst_list:
            for b in range(input.size()[0]):
                indices = (inst[b:b+1] == int(i)).nonzero() # n x 4            
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]]                    
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)                                        
                    outputs_mean[indices[:,0] + b, indices[:,1] + j, indices[:,2], indices[:,3]] = mean_feat                       
        return outputs_mean

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=1, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:                                
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D
        result = []
        input_downsampled = input
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result
        
# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw-1.0)/2)) #1
        # np.ceil(ndarray)计算大于等于该值的最小整数
        # >>> a = np.array([-1.7, -1.5, -0.2, 0.2, 1.5, 1.7, 2.0])
        # >>> np.ceil(a)
        # array([-1., -1., -0.,  1.,  2.,  2.,  2.])

        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]]

        nf = ndf
        for n in range(1, n_layers):
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw),
                norm_layer(nf), nn.LeakyReLU(0.2, True)
            ]]

        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:
            for n in range(len(sequence)):
                setattr(self, 'model'+str(n), nn.Sequential(*sequence[n]))
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]
            self.model = nn.Sequential(*sequence_stream)

    def forward(self, input):
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers+1):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)        

from torchvision import models
class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice1 = torch.nn.Sequential()
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(21, 30):
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)        
        h_relu3 = self.slice3(h_relu2)        
        h_relu4 = self.slice4(h_relu3)        
        h_relu5 = self.slice5(h_relu4)                
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
        return out
