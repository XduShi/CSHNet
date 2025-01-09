from audioop import bias
from modulefinder import Module
from turtle import forward
from cv2 import threshold
import torch
from torch._C import TracingState
import torch.nn as nn
import functools
from torch.autograd import Variable
import numpy as np
from torch.nn import modules
from torch.nn.modules import padding
import cv2
from util import util
from numba import jit
#from sobel.sobel_try2 import Threshold, pixcel
#from FGGAN.sota.CycleGAN.models.networks import PixelDiscriminator
from .layers import DropPath, to_2tuple, trunc_normal_
from einops import rearrange
import torch.nn.functional as F
import math
#from focal_frequency_loss import FocalFrequencyLoss as FFL

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

##定义PatchUnEmbed  用于卷积操作


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
######################--定义PoolFormer Block--###-edit 2021/11/26-################ 
##定义MLP
class Mlp(nn.Module):  # Mlp inside encoder
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_features, hidden_features, 1)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
##定义GroupNorm
class GroupNorm(nn.GroupNorm):
    def __init__(self, num_channels, **kwargs):
        super().__init__(1, num_channels, **kwargs)
##定义替换Attention的Pooling层
class Pooling(nn.Module):
    def __init__(self, pool_size = 3):
        super().__init__()
        self.pooling = nn.AvgPool2d(pool_size, stride=1, padding=pool_size//2,count_include_pad=False)
    def forward(self,x):
        return self.pooling(x) - x
##定义PoolFormerBlock
class PoolFormerBlock(nn.Module):
    def __init__(self, dim, pool_size=3, mlp_ratio=4.,act_layer=nn.GELU, norm_layer=GroupNorm,
                 drop=0., drop_path=0., use_layer_scale=True, layer_scale_init_value=1e-5):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.token_mixer = Pooling(pool_size=pool_size)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim*mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer,drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            self.layer_scale_1 = nn.Parameter(layer_scale_init_value*torch.ones((dim)),requires_grad=True)
            self.layer_scale_2 = nn.Parameter(layer_scale_init_value*torch.ones((dim)),requires_grad=True)
    def forward(self, x):
        if self.use_layer_scale:
            x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        else:
            x = x + self.drop_path(self.token_mixer(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x
            
#####################################################################################
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
    
    
##定义attention 采用Swin V2  ---a scales cosine attention (Sim(qi,kj)=cos(qi,kj)/tau + Bij)  ----edit 2022/3/1
class WindowAttentionV2(nn.Module):
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

    def __init__(self, dim, window_size, num_heads, meta_network_hidden_features=256, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5 #^(-1/2)

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH
    

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
        ##定义元网络 用于生成任意相对坐标的偏差值
        self.meta_network = nn.Sequential(
            nn.Linear(in_features=2, out_features=meta_network_hidden_features, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=meta_network_hidden_features, out_features=num_heads, bias=True)
        )
        ## init tau
        self.register_parameter("tau", torch.nn.Parameter(torch.ones(1, num_heads, 1, 1)))
        ## Init pair-wise relative positions (log-spaced)
        self.__make_pair_wise_relative_positions()
        
        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)
    def __make_pair_wise_relative_positions(self):
        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 将（Wh,Wh) 与 (Wh,Wh)在dim = 0维度合并->2, Wh, Ww -->2, 16, 16
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww 利用广播机制，分别在第一维，第二维，插入一个维度，进行广播相减 --> 2,256,256
        relative_coords = relative_coords.permute(1, 2, 0).reshape(-1,2).float()  # (Wh*Ww)*(Wh*Ww), 2 -->65536,2
        relative_coords_log = torch.sign(relative_coords) * torch.log(1. + relative_coords.abs())
        self.register_buffer("relative_coords_log", relative_coords_log) #不参与网络学习的变量
        
    def __get_relative_positional_encodings(self):
        """
        Method computes the relative positional encodings
        :return: (torch.Tensor) Relative positional encodings [1, number of heads, window size ** 2, window size ** 2]
        """
        relative_position_bias = self.meta_network(self.relative_coords_log)
        relative_position_bias = relative_position_bias.permute(1,0)
        relative_position_bias =relative_position_bias.reshape(self.num_heads, self.num_heads*self.num_heads,
                                                               self.num_heads*self.num_heads)
        return relative_position_bias.unsqueeze(0)    
        
    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) #--> (3, 16, 16, 256, 16)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple) #-->(16, 16, 256, 16)

        #q = q * self.scale
        #attn = (q @ k.transpose(-2, -1)) #--> (16,16,256,256)
        ###----scaled cosine attention---- cos(theta)=A*B/(||A||*||B||)
        #attn --> (16,16,256,256)  (q @ k.transpose(-2, -1)) --> (16,16,256,256)  torch.norm(q, dim=-1, keepdim=True) --> (16, 16, 256, 1)
        #torch.norm(k, dim=-1, keepdim=True).transpose(-2,-1) --> (16,16,1,256)
        ##余弦注意力
        attn = (q @ k.transpose(-2, -1)) / torch.maximum(torch.norm(q, dim=-1, keepdim=True)
                                                         * torch.norm(k, dim=-1, keepdim=True).transpose(-2,-1),
                                                         torch.tensor(1e-06, device=q.device, dtype=q.dtype))  ##torch.norm() 对输入求p范数，默认为2-范数
        attn = attn / self.tau.clamp(min=0.01)
        attn = attn + self.__get_relative_positional_encodings()

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


##定义feature-space local attention ---edit 2022/5/10
class ContentAttention(nn.Module):
    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., kmeans=False):
        super().__init__()
        self.dim = dim
        self.ws = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.kmeans = kmeans
        self.qkv = nn.Linear(dim, dim*3, bias = qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.get_v = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim)
        
    def forward(self, x, mask=None):
        B_, N, C = x.shape   #(1,64*64=4096,256)
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C//self.num_heads).permute(2,0,3,1,4)  #(3,1,16,4096,16)
        if True:
            q_pre = qkv[0].reshape(B_*self.num_heads, N, C//self.num_heads).permute(0,2,1)  #(16,16,4096)
            ntimes = int(math.log(N//(self.ws[0] * self.ws[1]), 2)) ##总共分组的次数=log(16)=4
            q_idx_last = torch.arange(N).cuda().unsqueeze(0).expand(B_*self.num_heads, N) ## 最后的聚合结果
            for i in range(ntimes):
                bh, d, n = q_pre.shape  #(16,16,4096)-->(32,16,2048)-->(64,16,1024)-->(128,16,512)
                q_pre_new = q_pre.reshape(bh,d,2,n//2) ##将所有tokens分为2组 (16,16,2,2048)-->(32,16,2,1024)-->(64,16,2,512)-->(128,16,2,256)
                q_avg = q_pre_new.mean(dim=-1) ##求两组tokens分别的均值 (bh,d,2)=(16,16,2)-->(32,16,2)-->(64,16,2)-->(128,16,2)
                q_avg = torch.nn.functional.normalize(q_avg, dim=-2) #d维l2范数的标准化后的数据 (16,16,2) 某一列的某个值 = 当前值/所处列的l2范数
                iters = 2 ##每一次划分后迭代寻找聚类质心的次数
                for i in range(iters):
                    q_scores = torch.nn.functional.normalize(q_pre.permute(0,2,1),dim=-1).bmm(q_avg) ##(bh,n,2)=(16,4096,2)-->(32,2048,2)-->(64,1024,2)-->(128,512,2)
                    soft_assign = torch.nn.functional.softmax(q_scores*100, dim=-1).detach()
                    q_avg = q_pre.bmm(soft_assign) ##(bh,d,2)=(16,16,2) -->(32,16,2)-->(64,16,2)-->(128,16,2)
                    q_avg = torch.nn.functional.normalize(q_avg, dim=-2)
                q_scores = torch.nn.functional.normalize(q_pre.permute(0,2,1), dim=-1).bmm(q_avg).reshape(bh, n, 2) #(16,4096,2)-->(32,2048,2)-->(64,1024,2)-->(128,512,2)
                q_idx = (q_scores[:,:,0]+1)/(q_scores[:,:,1]+1) ##相当于论文ri (16,4096)-->(32,2048)-->(64,1024)-->(128,512)
                _, q_idx = torch.sort(q_idx, dim=-1)
                q_idx_last = q_idx_last.gather(dim=-1, index=q_idx).reshape(bh*2, n//2)  #(32,2048)-->(64,1024)-->(128,512)-->(256,256)
                q_idx = q_idx.unsqueeze(1).expand(q_pre.size()) #(16,16,4096)-->(32,16,2048)-->(64,16,1024)-->(128,16,512)
                q_pre = q_pre.gather(dim=-1,index=q_idx).reshape(bh,d,2,n//2).permute(0,2,1,3).reshape(bh*2,d,n//2) #(32,16,2048)-->(64,16,1024)-->(128,16,512)-->(256,16,256)
                
            q_idx = q_idx_last.view(B_, self.num_heads, N) #(1,16,4096)
            _, q_idx_rev = torch.sort(q_idx, dim=-1)
            q_idx = q_idx.unsqueeze(0).unsqueeze(4).expand(qkv.size()) #(3,1,16,4096,16)
            qkv_pre = qkv.gather(dim=-2, index=q_idx)
            
            q, k, v = rearrange(qkv_pre, 'qkv b h (nw ws) c -> qkv (b nw) h ws c', ws=64) #(64,16,64,16)
            
            k = k.view(B_*((N//64))//2,2,self.num_heads, 64, -1) #(32,2,16,64,16)
            k_over1 = k[:,1,:,:20].unsqueeze(1) #(32,1,16,20,16)
            k_over2 = k[:,0,:,44:].unsqueeze(1) #(32,1,16,20,16)
            k_over = torch.cat([k_over1, k_over2],1) #(32,2,16,20,16)
            k = torch.cat([k, k_over], 3).contiguous().view(B_*((N//64)), self.num_heads, 64+20, -1) #(64,16,84,16)

            v = v.view(B_*((N//64))//2,2,self.num_heads, 64, -1)
            v_over1 = v[:,1,:,:20].unsqueeze(1)
            v_over2 = v[:,0,:,44:].unsqueeze(1)
            v_over = torch.cat([v_over1,v_over2],1)
            v = torch.cat([v, v_over], 3).contiguous().view(B_*((N//64)), self.num_heads, 64+20, -1) #(64, 16, 84, 16)
            
        attn = (q @ k.transpose(-2, -1))*self.scale #(64,16,64,84)
        attn = self.softmax(attn)
        attn = self.attn_drop(attn)
        out = attn @ v #(64,16,64,16)
        
        if True:
            out = rearrange(out, '(b nw) h ws d -> b (h d) nw ws', h = self.num_heads, b=B_)  ##(1,256,64,64)
            v = rearrange(v[:,:,:64,:], '(b nw) h ws d -> b h d (nw ws)', h = self.num_heads, b=B_) ##(1,16,16,64*64)
            W = int(math.sqrt(N)) #64
            ##相对位置编码
            out = out.reshape(B_, self.num_heads, C//self.num_heads, -1) #(1,16,16,4096)
            q_idx_rev = q_idx_rev.unsqueeze(2).expand(out.size()) #(1,16,16,4096)
            x = out.gather(dim=-1, index=q_idx_rev).reshape(B_,C,N).permute(0,2,1) #(1,4096,256)
            v = v.gather(dim=-1, index=q_idx_rev).reshape(B_,C,W,W) #(1,256,64,64)
            v = self.get_v(v) #(1,256,64,64)
            v = v.reshape(B_,C,N).permute(0,2,1) #(1,4096,256)
            x = x+v #(1,4096,256)
        
        x = self.proj(x)
        x = self.proj_drop(x)
        return x
###PatchMerging 用于合并Patch 相当于下采样

###定义下采样DownSampling 替换PatchMerging edit-2021-10-19


###PatchExpand用于上采样，扩大patch尺寸

#定义上采样UpSampling，替换PatchExpand edit-2021-10-19


###patch_size = 4 时dim_scale=4， nn.Linear(dim, 16*dim, bias=False)
###patch_size = 2 时dim_scale=2， nn.Linear(dim, 4*dim, bias=False)

#定义Final_up 最后一层上采样 替换 FinalPatchExpand_X4 edit-2021-10-19

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
        self.attn = WindowAttentionV2(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        
        
        ####添加feature-space-local-attention ---edit 2022/5/10 ---nonono so bad
        #if self.shift_size > 0:
        #    self.attnC = ContentAttention(dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
        #                                  qkv_bias=qkv_bias, qk_scale=qk_scale,attn_drop=attn_drop,
        #                                  proj_drop=drop, kmeans=shift_size)
        #    self.norm3 = norm_layer(dim)
        
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
        H, W = self.input_resolution #(64,64)
        #H = self.input_resolution
        #W = self.input_resolution
        B, L, C = x.shape #(1,64*64,256)
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        
        x = x.view(B, H, W, C) #(1,64,64,256)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C --> (16,16,16,256)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C --> (16, 16*16, 256)

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)  # nW*B, window_size*window_size, C -->(16,256,256)
        # merge windows
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C) # (16,16,16,256)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C --> (1,64,64,256)

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x   ## -->(1,64,64,256)
        x = x.view(B, H * W, C)  ## -->(1,4096,256)
        x = self.norm1(x) #(1,64*64,256)
        x = shortcut + self.drop_path(x)
        
        ###加入contentattention ---edit 2022/5/10 ---nonono
        #if self.shift_size > 0:
        #    x = x + self.norm3(self.attnC(x))
        #######################################
        # FFN
        x = x + self.drop_path(self.norm2(self.mlp(x)))

        return x

###位置编码### ---edit2022/5/14 用于低级特征学习
class CPosEnc(nn.Module):
    def __init__(self, dim, k=3, act=False, normtype=False):
        super().__init__()
        self.proj = nn.Conv2d(dim,
                              dim,
                              to_2tuple(k),
                              to_2tuple(1),
                              to_2tuple(k // 2),
                              groups=dim)
        self.normtype = normtype
        if self.normtype == 'batch':
            self.norm = nn.BatchNorm2d(dim)
        elif self.normtype == 'layer':
            self.norm = nn.LayerNorm(dim)
        self.activation = nn.GELU() if act else nn.Identity()

    def forward(self, x):
        B, N, C = x.shape
        H,W = int(math.sqrt(N)), int(math.sqrt(N))
        feat = x.transpose(1, 2).view(B, C, H, W)
        feat = self.proj(feat)
        if self.normtype == 'batch':
            feat = self.norm(feat).flatten(2).transpose(1, 2)
        elif self.normtype == 'layer':
            feat = self.norm(feat.flatten(2).transpose(1, 2))
        else:
            feat = feat.flatten(2).transpose(1, 2)
        x = x + self.activation(feat)
        return x

###定义空间注意力### ---edit 2022/5/14 用于低级特征学习 
class SpaceFormer(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4,drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, cpe_act=False):
        super().__init__()
        self.cpe = nn.ModuleList([CPosEnc(dim=dim,k=3,act = cpe_act),
                                  CPosEnc(dim=dim,k=3,act = cpe_act)])
        self.attn = nn.MultiheadAttention(dim, num_heads,dropout=0)
        self.norm = norm_layer(dim)
        self.mlp = MlpC(in_features=dim, hidden_features=dim*mlp_ratio, act_layer=act_layer,drop=drop_path)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x):
        x = self.cpe[0](x)
        cur = self.attn(x,x,x)[0]
        cur = self.norm(cur)
        x = x + self.drop_path(cur)
        x = self.cpe[1](x)
        x = x + self.drop_path(self.norm(self.mlp(x)))
        return x

###定义提取低级特征模块
class Low_feature(nn.Module):
    def __init__(self, input_nc, init_nc, dim, num_sf, num_heads):
        super().__init__()
        activation = nn.ReLU(True)
        norm_layer = nn.BatchNorm2d
        self.conv = nn.Sequential(nn.Conv2d(input_nc, init_nc, kernel_size=3, stride=1, padding=1), norm_layer(init_nc), activation)
        self.down = nn.Sequential(nn.Conv2d(init_nc, dim, kernel_size=8, stride=8), norm_layer(dim))
        spacefomer = []
        for i in range(num_sf): 
            spacefomer += [SpaceFormer(dim, num_heads, mlp_ratio=4, drop_path=0)]
        self.SpaceFormer = nn.Sequential(*spacefomer) 
        self.a = nn.Conv2d(dim,dim,kernel_size=1,stride=1)
        self.b = nn.Conv2d(dim,dim,kernel_size=1,stride=1)
    def forward(self, x):
        x = self.conv(x) #(1,64,256,256)
        x = self.down(x) #(1,256,64,64)
        B,C,H,W = x.shape
        x_t = x.permute(0,2,3,1).view(B,-1,C) #(1,4096,256)
        x_t = self.SpaceFormer(x_t) 
        x_t = x_t.view(B,H,W,C).permute(0,3,1,2) #(1,256,64,64)
        a = self.a(x_t)
        b = self.b(x_t)
        return a,b
###定义下采样层


#上采样层

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
def define_G(ngf, input_nc, output_nc, n_downsampling):
    
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
    ###采用LGG模型---edit 2022/4/29 + 2022/5/14
    netG = LGG(input_nc, output_nc, ngf, n_downsampling, n_blocks=4, num_heads=16, window_size=16)
    ###采用SwinG模型--用于对比(中间过度块全是SwinTransformer) ---edit 2022/5/30
    #netG = SwinG(input_nc, output_nc, ngf, n_downsampling, n_blocks=9, num_heads=16, window_size=16)
    ####采用ResTrans模型--edit2021/11/10
    #netG = ResTrans_EN(input_nc, output_nc, ngf, n_downsampling, n_blocks=4, num_heads=16, window_size=16)
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
    ##采用了空间注意力的判别器---edit 2022/6/2 ---不采用
    #netD = MultiD_wSF(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)   
    print(netD)
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
        for i in range(len(x_vgg)):  ## vgg[0]-->(1,64,256,256) vgg[1] -->(1,128,128,128) vgg[2] -->(1,256,64,64)  vgg[3] --> (1,512,32,32) vgg[4] -->(1,512,16,16)
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())  #t.detach() 将numpy转换为tensor   #tensor.detach() 返回一个新的tensor，从当前计算图中分离，但仍指向原变量的存放位置，requires_grad=false，不计算梯度
        return loss

#######利用sobel算子边缘检测设计损失 ---edit 2022/3/7##########################
    ###定义高低阈值
#统计像素值用于选择最低阈值
#def pixcel(img):
#    H, W = img.shape
#    hest = np.zeros([256],dtype = np.int32)
#    for i in range(1, H-1):
#        for j in range(1, W - 1):
            #统计不同像素值出现的频率
#            pv = img[i,j]
#            hest[pv] +=1
#            pix = hest.argsort()[-30:][::-1] #取像素出现频率前30的像素
#            num = max(pix) 
#            if num > 67: #防止最高阈值大于255
#                for m in range(len(pix)):
#                    if pix[m] > 67:
#                        pix[m] = 0
#                num = max(pix)
#    return num 
@jit
def max_entropy(img):
    def calcu_entropy(hist,threshold):
        sum_pixels=img.shape[0]*img.shape[1]
        temp_hist=hist/sum_pixels
        sum1=0
        sum2=0
        for i in range(256):
            if i<threshold:
                sum1+=temp_hist[i]
            else:
                sum2+=temp_hist[i]
        entropy1=0
        entropy2=0
        if (sum1==0)|(sum2==0):
            return 0
        for i in range(256):
            if i<=threshold:
                if temp_hist[i]!=0:
                    entropy1-=(temp_hist[i]/sum1)*np.log2(temp_hist[i]/sum1)
            else:
                if temp_hist[i]!=0:
                    entropy2-=(temp_hist[i]/sum2)*np.log2(temp_hist[i]/sum2)
        entropy=entropy1+entropy2
        return entropy
    def max_Entropy(img):
        max_ent=0.
        max_index=0
        #hist=cv2.calcHist(img,[0],None,[256],[0,255])#计算灰度直方图  返回一维数组
        hist, bins = np.histogram(img.ravel(), 256, [0, 256])
        #print(hist)
        for i in range(256):
            entropy=calcu_entropy(hist,i)
            if entropy>max_ent:
                max_ent=entropy
                max_index=i
        return max_index
    tl=max_Entropy(img)
    if tl > 90:
        tl = tl // 2.5
    return tl

    
# get edge strength and edge angle
def get_edge_angle(fx, fy):
    # get edge strength
    edge = np.sqrt(np.power(fx.astype(np.float32), 2) + np.power(fy.astype(np.float32), 2))
    edge = np.clip(edge, 0, 255)
 
    # make sure the denominator is not 0
    fx = np.maximum(fx, 1e-10)
    #fx[np.abs(fx) <= 1e-5] = 1e-5
 
    # get edge angle
    angle = np.arctan(fy / fx)
 
    return edge, angle
 
    
# 将角度量化为0°、45°、90°、135°
def angle_quantization(angle):
    angle = angle / np.pi * 180
    angle[angle < -22.5] = 180 + angle[angle < -22.5]
    _angle = np.zeros_like(angle, dtype=np.uint8)
    _angle[np.where(angle <= 22.5)] = 0
    _angle[np.where((angle > 22.5) & (angle <= 67.5))] = 45
    _angle[np.where((angle > 67.5) & (angle <= 112.5))] = 90
    _angle[np.where((angle > 112.5) & (angle <= 157.5))] = 135
 
    return _angle
 
@jit 
def non_maximum_suppression(angle, edge):
    H, W = angle.shape
    _edge = edge.copy()
    
    for y in range(H):
        for x in range(W):
                if angle[y, x] == 0:
                    dx1, dy1, dx2, dy2 = -1, 0, 1, 0
                elif angle[y, x] == 45:
                    dx1, dy1, dx2, dy2 = -1, 1, 1, -1
                elif angle[y, x] == 90:
                    dx1, dy1, dx2, dy2 = 0, -1, 0, 1
                elif angle[y, x] == 135:
                    dx1, dy1, dx2, dy2 = -1, -1, 1, 1
                # 边界处理
                if x == 0:
                    dx1 = max(dx1, 0)
                    dx2 = max(dx2, 0)
                if x == W-1:
                    dx1 = min(dx1, 0)
                    dx2 = min(dx2, 0)
                if y == 0:
                    dy1 = max(dy1, 0)
                    dy2 = max(dy2, 0)
                if y == H-1:
                    dy1 = min(dy1, 0)
                    dy2 = min(dy2, 0)
                # 如果不是最大值，则将这个位置像素值置为0
                if max(max(edge[y, x], edge[y + dy1, x + dx1]), edge[y + dy2, x + dx2]) != edge[y, x]:
                    _edge[y, x] = 0
 
    return _edge

           
#定义双阈值
#def Threshold(img, threshold1, threshold2):
#    H, W = img.shape
#    TL = min(threshold1, threshold2) #最低阈值
#    TH = max(threshold1, threshold2) #最高阈值
#    DT = np.zeros([H,W]) #初始化阈值图
#    for i in range(1, H - 1):
#        for j in range(1, W - 1):
#            #小于TL的设置为0
#            if img[i,j] < TL:
#                DT[i,j]=0
#            #大于TH的值设置为255，即边缘
#            elif img[i,j] > TH:
#                DT[i,j] = 255
            #对于在TL-TH之间的数，如果相邻点大于TH，及和边缘值有链接，则也设为边缘值255
#            elif img[i-1,j]>TH or img[i-1,j-1]>TH or img[i-1,j+1]>TH or img[i,j-1]>TH or img[i,j+1]>TH or img[i+1,j]>TH or img[i+1,j-1]>TH or img[i+1,j+1]>TH:
#                DT[i,j]=255
#            else:
#                DT[i,j] = img[i,j]    
#    return DT

#定义双阈值
def Threshold(img, threshold1, threshold2):
    TL = min(threshold1, threshold2) #最低阈值
    TH = max(threshold1, threshold2) #最高阈值
    img[img<TL]=0
    img[img>TH]=255
    return img

class SobelLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(SobelLoss, self).__init__()
        self.criterion = nn.MSELoss()
        
        
    def forward(self, x, y):
        ##对真实输入图像进行处理(1,3,256,256)-->(256,256,3)
        #x_p = x.squeeze(0)
        #x_p = x_p.permute(1, 2, 0)
        #x_p = x_p.cpu().numpy()*255
        x_p = x.mul(255).byte()
        x_p = x_p.cpu().numpy().squeeze(0).transpose((1,2,0))
        ##对生成输入图像进行处理(1,3,256,256)-->(256,256,3)
        #y_p = y.squeeze(0)
        #y_p = y_p.permute(1, 2, 0)
        #y_p = y_p.cpu().detach().numpy()*255
        y_p = y.mul(255).byte()
        y_p = y_p.cpu().numpy().squeeze(0).transpose((1,2,0))
        ##进行高斯滤波
        x_G =  cv2.GaussianBlur(x_p,(9,9),0)
        y_G =  cv2.GaussianBlur(y_p,(9,9),0)
        ##灰度化处理
        gray1 = cv2.cvtColor(x_G, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(y_G, cv2.COLOR_BGR2GRAY)
        ##利用Sobel算子边缘检测
        #真实图像
        real_x = cv2.Sobel(gray1, cv2.CV_16S, 1, 0)  # 对x求一阶导 ##cv2.cv_16s： 16位有符号整数
        real_y = cv2.Sobel(gray1, cv2.CV_16S, 0, 1)  # 对y求一阶导
        abs_real_x = cv2.convertScaleAbs(real_x)
        abs_real_y = cv2.convertScaleAbs(real_y)
        #real_o = cv2.addWeighted(abs_real_x, 0.5, abs_real_y, 0.5, 0)
        edge1, angle1 = get_edge_angle(abs_real_x, abs_real_y)
        angle1 = angle_quantization(angle1)
        real_o = non_maximum_suppression(angle1, edge1)
        ##定义阈值
        threshold1 = max_entropy(real_o) 
        threshold2 = 180
        #经过阈值的图像
        DT_real =Threshold(real_o, threshold1=threshold1, threshold2=threshold2)
        real = torch.autograd.Variable(torch.from_numpy(DT_real).float(), requires_grad=True).cuda()
        
        #生成图像
        fake_x = cv2.Sobel(gray2, cv2.CV_16S, 1, 0)  # 对x求一阶导 ##cv2.cv_16s： 16位有符号整数
        fake_y = cv2.Sobel(gray2, cv2.CV_16S, 0, 1)  # 对y求一阶导
        abs_fake_x = cv2.convertScaleAbs(fake_x)
        abs_fake_y = cv2.convertScaleAbs(fake_y)
        #fake_o = cv2.addWeighted(abs_fake_x, 0.5, abs_fake_y, 0.5, 0)
        
        edge2, angle2 = get_edge_angle(abs_fake_x, abs_fake_y)
        angle2 = angle_quantization(angle2)
        fake_o = non_maximum_suppression(angle2, edge2)
        DT_fake =Threshold(fake_o, threshold1=threshold1, threshold2=threshold2)
        fake = torch.autograd.Variable(torch.from_numpy(DT_fake).float(), requires_grad=True).cuda()
        ##计算loss
        loss = self.criterion(real, fake) * 0.01
        return loss
        
#############################################################################
#ssim loss

##############################################################################
# Generator
##############################################################################

###########定义含有Swin-Transformer的生成器#######

    

##运用p2pHD的思想----edit 2021/10/28

       

##探索Transformer和residual block的关系 将residual block 替换为swin block 利用GlobalG架构--edit2021/11/2

    



##将decoder改为卷积 效果很差


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

  ####设置可学习参数，让网络自动学习CNN和Transformer的比例-----edit 2022-3-4-------
class L_ResSA(nn.Module):
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
        
        ###设置两个可学习的参数w1, w2 初始化均为0.5，0.5 ---只考虑了CNN和Transformer，没有考虑跳跃连接的影响 --edit 2022-3-4--
        ##设置3个可学习参数w0,w1,w2 w0--跳跃连接   w1--CNN  w2--Transformer 均初始化为0.33 --edit 2022-3-10---
        #设置2个可学习参数w1,w2 初始化均为0.5----考虑跳跃连接和CNN-Transformer的cat结果
        #self.w0 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w1 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(1), requires_grad=True)
        #初始化
        #self.w0.data.fill_(0.33)
        self.w1.data.fill_(0.5)
        self.w2.data.fill_(0.5)
    def forward(self,x):
        ##分支1经过卷积提取局部细节信息
        x1 = self.Res_Block(x)
        ##分支2经过Transformer建立长程依赖关系
        B, C, H, W = x.shape
        x2 = x.permute(0,2,3,1).view(B,-1,C)
        x2 = self.Swin_Block(x2)
        x2 = x2.view(B, H, W, C).permute(0,3,1,2)
        ##两(三）个分支按照可学习的参数进行像素级相加
        #保证两(三)个参数均大于0且和为1
        #w0_relu = torch.nn.functional.relu(self.w0)
        w1_relu = torch.nn.functional.relu(self.w1)
        w2_relu = torch.nn.functional.relu(self.w2)
        #w0 = (w0_relu / (w0_relu + w1_relu + w2_relu))
        #w1 = (w1_relu / (w0_relu + w1_relu + w2_relu))
        #w2 = (w2_relu / (w0_relu + w1_relu + w2_relu))
        w1 = (w1_relu / (w1_relu + w2_relu))
        w2 = (w2_relu / (w1_relu + w2_relu))
        #x3 = w1 * x1 + w2 * x2 
        #out = x + x3       
        #out = w0 * x + w1 * x1 + w2 *x2
        ##两个分支合并
        x3 = torch.cat([x1,x2],1)
        ##降维
        x4 = self.conv(x3)
        out = w1*x + w2*x4
        return out
    
#######重新设计CNN与Transformer相结合的模块 --edit 2022/4/29 ###############################
###定义local_block:卷积为主，在经过Swin后（全局）中更突出细节
class Local_Block(nn.Module):
    def __init__(self, dim, activation = nn.ReLU(True), num_heads = 16, window_size = 16):
        super().__init__()
        #定义卷积模块
        norm_layer1 = nn.BatchNorm2d
        cnn_block = []
        cnn_block += [nn.Conv2d(dim,dim,kernel_size=1,stride=1), nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1),
                      nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1), nn.Conv2d(dim,dim,kernel_size=1,stride=1),
                      norm_layer1(dim), activation]
        cnn_block += [nn.Conv2d(dim,dim,kernel_size=1,stride=1), nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1),
                      nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1), nn.Conv2d(dim,dim,kernel_size=1,stride=1),
                      norm_layer1(dim)]
        self.CNN_Block = nn.Sequential(*cnn_block)
        ###定义swin block
        swin_block = []
        ngf = 64
        for i in range(2):
            swin_block +=[SwinTransformerBlock(dim, input_resolution=(64,64),
                         num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=4.0,
                                 qkv_bias=True, qk_scale=None,
                                 drop=0.0, attn_drop=0.0,
                                 drop_path=0.2,
                                 norm_layer=nn.LayerNorm)]
        self.Swin_Block = nn.Sequential(*swin_block)
        ###1*1conv降维
        self.dim_d = nn.Conv2d(2*dim, dim, kernel_size=1, stride=1)
        ###定义激活sigmoid
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_CNN = self.CNN_Block(x) + x  ##加入残差连接 (B,C,H,W)
        x_swin = x_CNN.permute(0,2,3,1).view(B,-1,C)  ##(B, HW, C)
        x_swin = self.Swin_Block(x_swin) ##(B,HW,C)
        x_Swin = x_swin.view(B, H, W, C).permute(0,3,1,2) ##(B,C,H,W)
        ###定义激活(权值)
        w_cnn = self.sigmoid(x_CNN)
        ###在全局特征x_Swin中再次加强局部细节信息
        x_Swin_E = w_cnn*x_Swin
        ###将x_CNN与经过transformer的全局特征图(经过局部细节增强的)cat一起
        x_CS = torch.cat([x_CNN, x_Swin_E], 1)
        ###降维
        out = self.dim_d(x_CS)
        return out

class SEC(nn.Module):
    def __init__(self, dim, activation = nn.ReLU(True), num_heads = 16, window_size = 16):
        super().__init__()
        #CNN Block
        norm_layer1 = nn.BatchNorm2d
        cnn_block = []
        cnn_block += [nn.Conv2d(dim,dim,kernel_size=1,stride=1), nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1),
                      nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1), nn.Conv2d(dim,dim,kernel_size=1,stride=1),
                      norm_layer1(dim), activation]
        cnn_block += [nn.Conv2d(dim,dim,kernel_size=1,stride=1), nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1),
                      nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1), nn.Conv2d(dim,dim,kernel_size=1,stride=1),
                      norm_layer1(dim)]
        self.CNN_Block = nn.Sequential(*cnn_block)
        #Swin Block
        swin_block = []
        ngf = 64
        for i in range(2):
            swin_block +=[SwinTransformerBlock(dim, input_resolution=(64,64),
                         num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2, mlp_ratio=4.0, qkv_bias=True, 
                                 qk_scale=None,drop=0.0, attn_drop=0.0, drop_path=0.2, norm_layer=nn.LayerNorm)]
        self.Swin_Block = nn.Sequential(*swin_block)
        ###1*1conv
        self.dim_d = nn.Conv2d(2*dim, dim, kernel_size=1, stride=1)
        ###sigmoid
        self.sigmoid = nn.Sigmoid()       
    def forward(self, x):
        B, C, H, W = x.shape
        x_CNN = self.CNN_Block(x) + x  ## (B,C,H,W)
        x_swin = x_CNN.permute(0,2,3,1).view(B,-1,C)  ##(B, HW, C)
        x_swin = self.Swin_Block(x_swin) ##(B,HW,C)
        x_Swin = x_swin.view(B, H, W, C).permute(0,3,1,2) ##(B,C,H,W)
        ###weights
        w_cnn = self.sigmoid(x_CNN)
        x_Swin_E = w_cnn*x_Swin
        x_CS = torch.cat([x_CNN, x_Swin_E], 1)
        out = self.dim_d(x_CS)
        return out

class CES(nn.Module):
    def __init__(self, dim, activation = nn.ReLU(True), num_heads = 16, window_size = 16):
        super().__init__()
        #CNN Block
        norm_layer1 = nn.BatchNorm2d
        cnn_block = []
        cnn_block += [nn.Conv2d(dim,dim,kernel_size=1,stride=1), nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1),
                      nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1), nn.Conv2d(dim,dim,kernel_size=1,stride=1),
                      norm_layer1(dim), activation]
        cnn_block += [nn.Conv2d(dim,dim,kernel_size=1,stride=1), nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1),
                      nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1), nn.Conv2d(dim,dim,kernel_size=1,stride=1),
                      norm_layer1(dim)]
        self.CNN_Block = nn.Sequential(*cnn_block)
        #Swin Block
        swin_block = []
        for i in range(2):
            swin_block +=[SwinTransformerBlock(dim, input_resolution=(64,64),
                         num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2, mlp_ratio=4.0, qkv_bias=True, 
                                 qk_scale=None, drop=0.0, attn_drop=0.0, drop_path=0.2, norm_layer=nn.LayerNorm)]
        self.Swin_Block = nn.Sequential(*swin_block)
        ###1*1conv
        self.dim_d = nn.Conv2d(2*dim, dim, kernel_size=1, stride=1)
        ###sigmoid
        self.sigmoid = nn.Sigmoid()        
    def forward(self, x):
        B, C, H, W = x.shape
        x_swin = x.permute(0,2,3,1).view(B,-1,C)  ##(B, HW, C)
        x_swin = self.Swin_Block(x_swin) ##(B,HW,C)
        x_Swin = x_swin.view(B, H, W, C).permute(0,3,1,2) ##(B,C,H,W)
        x_CNN = self.CNN_Block(x_Swin) + x_Swin ##CNN
        ###weights
        w_swin = self.sigmoid(x_Swin)
        x_CNN_E = w_swin*x_CNN
        x_SC = torch.cat([x_Swin, x_CNN_E], 1)
        out = self.dim_d(x_SC)
        return out        


        
###定义global_block:transformer为主，在经过CNN后（局部）中更突出整体的把控
class Global_Block(nn.Module):
    def __init__(self, dim, activation = nn.ReLU(True), num_heads = 16, window_size = 16):
        super().__init__()
        #定义卷积模块
        norm_layer1 = nn.BatchNorm2d
        cnn_block = []
        cnn_block += [nn.Conv2d(dim,dim,kernel_size=1,stride=1), nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1),
                      nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1), nn.Conv2d(dim,dim,kernel_size=1,stride=1),
                      norm_layer1(dim), activation]
        cnn_block += [nn.Conv2d(dim,dim,kernel_size=1,stride=1), nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1),
                      nn.Conv2d(dim,dim,kernel_size=3,stride=1,padding=1), nn.Conv2d(dim,dim,kernel_size=1,stride=1),
                      norm_layer1(dim)]
        self.CNN_Block = nn.Sequential(*cnn_block)
        ###定义swin block
        swin_block = []
        ngf = 64
        for i in range(2):
            swin_block +=[SwinTransformerBlock(dim, input_resolution=(64,64),
                         num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=4.0,
                                 qkv_bias=True, qk_scale=None,
                                 drop=0.0, attn_drop=0.0,
                                 drop_path=0.2,
                                 norm_layer=nn.LayerNorm)]
        self.Swin_Block = nn.Sequential(*swin_block)
        ###1*1conv降维
        self.dim_d = nn.Conv2d(2*dim, dim, kernel_size=1, stride=1)
        ###定义激活sigmoid
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        B, C, H, W = x.shape
        x_swin = x.permute(0,2,3,1).view(B,-1,C)  ##(B, HW, C)
        x_swin = self.Swin_Block(x_swin) ##(B,HW,C)
        x_Swin = x_swin.view(B, H, W, C).permute(0,3,1,2) ##(B,C,H,W)
        x_CNN = self.CNN_Block(x_Swin) + x_Swin ##经过CNN
        ###定义激活(权值)
        w_swin = self.sigmoid(x_Swin)
        ###在局部特征x_CNN中再次强调全局的长程依赖性
        x_CNN_E = w_swin*x_CNN
        ###将x_CNN与经过transformer的全局特征图(经过局部细节增强的)cat一起
        x_SC = torch.cat([x_Swin, x_CNN_E], 1)
        ###降维
        out = self.dim_d(x_SC)
        return out        
        
            
############################################################################################################        
                

##定义含有Pool Former的下采样--2021/11/26
class DS(nn.Module):
    def __init__(self, in_dim, out_dim, norm_layer, activation):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.norm_layer = norm_layer
        self.activation = activation
        self.down = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.pool = PoolFormerBlock(out_dim)
    def forward(self,x):
        x = self.down(x)
        x = self.norm_layer(x)
        x = self.activation(x)
        x = self.pool(x)
        return x
####定义新的下采样模块，结合transformer---edit2022/4/7---
class En_DS(nn.Module):
    def __init__(self, in_dim, out_dim,activation, norm_layer=nn.LayerNorm, window_size = 16, num_heads=32, nhead=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., act_layer=nn.GELU, drop=0., dropout=0.1):
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.activation = activation
        self.window_size = window_size
        self.down_cnn = nn.Conv2d(in_dim, out_dim, kernel_size=3, stride=2, padding=1)
        self.norm_layer1 = norm_layer(in_dim)
        self.norm_layer2 = norm_layer(out_dim)
        self.dropout = nn.Dropout(dropout)
        #self.attn = WindowAttentionV2(in_dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
        #    qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=proj_drop)
        self.pool = PoolFormerBlock(in_dim)
        self.self_attn = nn.MultiheadAttention(in_dim, num_heads, dropout=0.1)
        self.multihead_attn = nn.MultiheadAttention(out_dim, num_heads, dropout=0.1)
        self.mlp1 = MlpC(in_features=in_dim, hidden_features=4*in_dim, act_layer=act_layer, drop=drop)
        self.mlp2 = MlpC(in_features=out_dim, hidden_features=4*out_dim, act_layer=act_layer, drop=drop)
    def forward(self,x): #x(1,64,256,256)
        B,C,H,W = x.shape
        x_cnn = self.down_cnn(x)  #(1,128,128,128)
        B1,C1,H1,W1 = x_cnn.shape
        x_= x.view(B,C,-1).permute(0,2,1) #(1,256*256,64)
        x_attn = self.self_attn(x_, x_, x_)[0]
        x_attn = self.norm_layer1(x_ +self.dropout(x_attn)) ##后归一化
        x_attn = self.norm_layer1(x_attn + self.dropout(self.mlp1(x_attn))) #(1,256*256,64)
        x_attn = x_attn.permute(0,2,1).view(B,C,H,W) #(1,64,256,256)
        #x_attn = self.pool(x)
        #x_attn = self.norm_layer(x_attn) + x_
        #x_attn = x_attn + self.norm_layer(self.mlp(x_attn))
        x_attn_d = self.down_cnn(x_attn)  #(1,128,128,128)
        #x_attn_d = x_attn_d.permute(0,2,1).view(B,C,H,W)  #(1,128,128,128)
        x_cnn_q = x_cnn.view(B1,C1,-1).permute(0,2,1)
        x_attn_kv = x_attn_d.view(B1,C1,-1).permute(0,2,1)
        attn = self.multihead_attn(x_cnn_q, x_attn_kv,x_attn_kv)[0]
        attn = self.norm_layer2(attn + self.dropout(attn))
        attn = self.norm_layer2(attn + self.dropout(self.mlp2(attn)))
        attn_ = attn.permute(0,2,1).view(B1,C1,H1,W1)
        return attn_
          
        
        
        
        
##定义上采样--2021/11/26
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
##定义包含ResSAR模块的生成器
class ResTrans(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=2, n_blocks=4, num_heads=16, window_size=16, scales=4):
        super().__init__()
        activation = nn.ReLU(True)
        norm_layer1 = nn.BatchNorm2d
        ##对SAR进行去噪
        #self.model0 = DeNoise(input_nc, ngf, norm=norm_layer1)     
        ##1*64*256*256 ##-edit 2021/11/26 下采样中添加Transformer(PoolFormerBlock)
        self.model1 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer1(ngf), activation, 
                                    PoolFormerBlock(ngf))
        ##down_sampling
        model_d = []
        for i in range(n_downsampling):
            mult = 2**i
            model_d += [DS(ngf*mult, ngf*mult*2, norm_layer1(mult*ngf*2), activation)]
            #model_d += [En_DS(ngf*mult, ngf*mult*2,activation, norm_layer=nn.LayerNorm, window_size = 8, num_heads=8, nhead=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., act_layer=nn.GELU, drop=0.)]
        self.model_D = nn.Sequential(*model_d)
        ##中间过渡层，采用ResSAR模块
        model_m = []
        mult = 2**n_downsampling
        for i in range(n_blocks):
            #n_blocks=4 记得在定义netG处修改n_blocks
            model_m += [ResSA(mult*ngf, activation=activation, num_heads=num_heads, window_size=window_size)]
        self.model_M = nn.Sequential(*model_m)
        #print(self.model_M.state_dict().keys())
        ##up_sampling
        model_u = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling-i)
            model_u += [US(ngf*mult, int(ngf*mult/2), norm_layer1(int(ngf*mult/2)), activation)]
        self.model_U = nn.Sequential(*model_u)
        self.model2 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh())
        ##concat 降维 --2021/11/26 concat两层 128*128*128 256*256*64
        self.concat_dim = nn.ModuleList()
        for i in range(n_downsampling): ## 0 1
            dim = ngf * 2**(n_downsampling - i)
            concat = nn.Conv2d(dim, int(dim/2), kernel_size=3, stride=1, padding=1)
            self.concat_dim.append(concat)  ##(256 --> 128) (128 --> 64)
################################################################################################################
#########---Cat1---#############################################################################################
    ##定义Encoder和用于concat的特征图
    def forward_features(self, x):
        x_downsample = []
        x = self.model1(x)
        x_downsample.append(x)
        for layer in self.model_D:
            x = layer(x)
            x_downsample.append(x)
        return x, x_downsample 
       
    ##定义采用concat特征图的Decoder
    def forward_up(self,x, x_downsample):
        for inx, layer_up in enumerate(self.model_U):
            x = layer_up(x)
            x = torch.cat([x, x_downsample[1-inx]], 1)
            x = self.concat_dim[inx](x)
        return x
    #############################################################################################################
    ##定义采用多感受野的cat方式
    #定义中间特征图
    def middle(self, x_downsample):
        x0 = x_downsample[0]  #(1,64,256,256)
        x1 = x_downsample[1]  #(1,128,128,128)
        x2 = x_downsample[2]  #(1,256,64,64)
        x_middle = []   ###[x1_m, x0_m2, x0_m1]
        for inx, layer_up in enumerate(self.model_U):
            x2 = layer_up(x2) # inx=0 (1,128,128,128)=x1_m   inx=1 (1,64,256,256)=x0_m2  
            x_middle.append(x2)
            if inx == 1:
                x0_m1 = layer_up(x1)
                x_middle.append(x0_m1)  #inx=1 (1,64,256,256)=x0_m1
        return x_middle
    #定义采用多尺度cat的Decoder
    def forward_up_cat(self, x, x_downsample, x_middle):
        x0 = x_downsample[0]
        x1 = x_downsample[1]
        x2 = x_downsample[2]
        for inx, layer_up in enumerate(self.model_U):
            x = layer_up(x)
            if inx == 0:
                x1_m = torch.cat([x1,x_middle[0]], 1)  #(1,256,128,128)
                x1_m = self.concat_dim[inx](x1_m)  #(1,128,128,128)
                x = torch.cat([x1_m, x], 1)   #(1,256,128,128)
                x = self.concat_dim[inx](x)  # (1,128,128,128)
            else:
                x0_m1 = torch.cat([x0, x_middle[2]], 1) #(1,128,256,256)
                x0_m1 = self.concat_dim[inx](x0_m1)   #(1,64,256,256)
                x0_m2 = torch.cat([x0_m1, x_middle[1]], 1)  #(1,128,256,256)
                x0_m2 = self.concat_dim[inx](x0_m2)   #(1,64,256,256)
                x = torch.cat([x0_m2, x], 1)  #(1,128,256,256)
                x = self.concat_dim[inx](x)   #(1,64,256,256)
        return x
            
    ########################################################################################
                    
    def forward(self,x):
        ##去噪
        #x = self.model0(x)
        
        ####采用cat1
        x, x_downsample = self.forward_features(x)   #x_downsample [(1,64,256,256) (1,128,128,128) (1,256,64,64)]
        x = self.model_M(x)
        x = self.forward_up(x, x_downsample)
        #x_middle = self.middle(x_downsample)  ##用于新cat方式
        #x = self.forward_up_cat(x,x_downsample,x_middle)  ##用于新cat方式
        x = self.model2(x)
        ####不采用cat
        #x = self.model1(x) #1*64*256*256
        #x = self.model_D(x) 
        #x = self.model_M(x) #1*256*64*64
        #x = self.model_U(x) 
        #x = self.model2(x)  #1*3*256*256
        return x

##定义包含Local_Block和Global_Block模块的生成器
class LGG(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=2, n_blocks=4, num_heads=16, window_size=16, scales=4):
        super().__init__()
        activation = nn.ReLU(True)
        norm_layer1 = nn.BatchNorm2d    
        ##1*64*256*256 ##-edit 2021/11/26 下采样中添加Transformer(PoolFormerBlock)
        self.model1 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer1(ngf), activation, 
                                    PoolFormerBlock(ngf))
        ###验证PoolFormer的作用 去掉它在model1和model_D中 ---edit2022/6/11
        #self.model1 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer1(ngf), activation)
        ##down_sampling
        model_d = []
        for i in range(n_downsampling+1):
            mult = 2**i
            model_d += [DS(ngf*mult, ngf*mult*2, norm_layer1(mult*ngf*2), activation)]
            #model_d += [En_DS(ngf*mult, ngf*mult*2,activation, norm_layer=nn.LayerNorm, window_size = 8, num_heads=8, nhead=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0., act_layer=nn.GELU, drop=0.)]
        self.model_D = nn.Sequential(*model_d)
        ##中间过渡层，采用ResSAR模块
        model_m = []
        mult = 2**(n_downsampling+1)
        ###2*Local_block+2*Global_block
        for i in range(n_blocks):
            #n_blocks=4 记得在定义netG处修改n_blocks
            if i % 2 == 0:
                model_m += [Local_Block(mult*ngf, activation=activation, num_heads=num_heads, window_size=window_size)]
            else:
                model_m += [Global_Block(mult*ngf, activation=activation, num_heads=num_heads, window_size=window_size)]
        self.model_M = nn.Sequential(*model_m)
        
        ###4*Local_block---用于验证
        # for i in range(n_blocks):
        #     model_m += [Local_Block(mult*ngf, activation=activation, num_heads=num_heads, window_size=window_size)]
        # self.model_M = nn.Sequential(*model_m)
        
        ###4*Global_block---用于验证
        # for i in range(n_blocks):
        #     model_m += [Global_Block(mult*ngf, activation=activation, num_heads=num_heads, window_size=window_size)]
        # self.model_M = nn.Sequential(*model_m)
        
        #print(self.model_M.state_dict().keys())
        ##up_sampling
        model_u = []
        for i in range(n_downsampling+1):
            mult = 2**(n_downsampling+1-i)
            model_u += [US(ngf*mult, int(ngf*mult/2), norm_layer1(int(ngf*mult/2)), activation)]
        self.model_U = nn.Sequential(*model_u)
        self.model2 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh())
        ###定义低级特征指导过渡层---edit 2022/5/14###
        self.low_feature = Low_feature(input_nc, init_nc=ngf, dim=ngf*8, num_sf=3, num_heads=num_heads)
        ######---edit 2022/5/22 利用attention产生a，b####
        ###定义互注意力 ---edit 2022/5/22
        #self.cross_attn = nn.MultiheadAttention(ngf*4, num_heads=num_heads,dropout=0)
        ###定义conv产生x_d-->q1, x_m2-->q2 ---edit 2022/5/22
        #self.q1 = nn.Conv2d(ngf*4, ngf*4, kernel_size=1, stride=1)
        #self.q2 = nn.Conv2d(ngf*4, ngf*4, kernel_size=1, stride=1)
        ##concat 降维 --2021/11/26 concat两层 128*128*128 256*256*64
        #self.concat_dim = nn.ModuleList()
        #for i in range(n_downsampling): ## 0 1
        #    dim = ngf * 2**(n_downsampling - i)
        #    concat = nn.Conv2d(dim, int(dim/2), kernel_size=3, stride=1, padding=1)
        #    self.concat_dim.append(concat)  ##(256 --> 128) (128 --> 64)
################################################################################################################
#########---Cat1---#############################################################################################
    ##定义Encoder和用于concat的特征图
    def forward_features(self, x):
        x_downsample = []
        x = self.model1(x)
        x_downsample.append(x)
        for layer in self.model_D:
            x = layer(x)
            x_downsample.append(x)
        return x, x_downsample 
       
    ##定义采用concat特征图的Decoder
    def forward_up(self,x, x_downsample):
        for inx, layer_up in enumerate(self.model_U):
            x = layer_up(x)
            x = torch.cat([x, x_downsample[1-inx]], 1)
            x = self.concat_dim[inx](x)
        return x
    #############################################################################################################
    ##定义采用多感受野的cat方式
    #定义中间特征图
    def middle(self, x_downsample):
        x0 = x_downsample[0]  #(1,64,256,256)
        x1 = x_downsample[1]  #(1,128,128,128)
        x2 = x_downsample[2]  #(1,256,64,64)
        x_middle = []   ###[x1_m, x0_m2, x0_m1]
        for inx, layer_up in enumerate(self.model_U):
            x2 = layer_up(x2) # inx=0 (1,128,128,128)=x1_m   inx=1 (1,64,256,256)=x0_m2  
            x_middle.append(x2)
            if inx == 1:
                x0_m1 = layer_up(x1)
                x_middle.append(x0_m1)  #inx=1 (1,64,256,256)=x0_m1
        return x_middle
    #定义采用多尺度cat的Decoder
    def forward_up_cat(self, x, x_downsample, x_middle):
        x0 = x_downsample[0]
        x1 = x_downsample[1]
        x2 = x_downsample[2]
        for inx, layer_up in enumerate(self.model_U):
            x = layer_up(x)
            if inx == 0:
                x1_m = torch.cat([x1,x_middle[0]], 1)  #(1,256,128,128)
                x1_m = self.concat_dim[inx](x1_m)  #(1,128,128,128)
                x = torch.cat([x1_m, x], 1)   #(1,256,128,128)
                x = self.concat_dim[inx](x)  # (1,128,128,128)
            else:
                x0_m1 = torch.cat([x0, x_middle[2]], 1) #(1,128,256,256)
                x0_m1 = self.concat_dim[inx](x0_m1)   #(1,64,256,256)
                x0_m2 = torch.cat([x0_m1, x_middle[1]], 1)  #(1,128,256,256)
                x0_m2 = self.concat_dim[inx](x0_m2)   #(1,64,256,256)
                x = torch.cat([x0_m2, x], 1)  #(1,128,256,256)
                x = self.concat_dim[inx](x)   #(1,64,256,256)
        return x
            
    ########################################################################################
                    
    #######采用低级特征指导过渡层 ---edit 2022/5/14 (a,b只用在了x_d和x_m2:效果还不错）######
    ###ax+b### ###验证1.ax 2.b####
    def low_mid(self,x,a,b):
        x_d = a*x + b
        #x_d = a*x   ###x_d = a*x+b(完整) 验证1.ax 2.x+b
        #x_d = x + b
        x_m1 = self.model_M[0](x_d)
        #x_m1 = a*x_m1+b ##2022/5/17
        x_m2 = self.model_M[1](x_m1)
        x_m2 = a*x_m2 + b
        #x_m2 = a*x_m2
        #x_m2 = x_m2 + b
        x_m3 = self.model_M[2](x_m2)
        #x_m3 = a*x_m3+b ##2022/5/17
        x_m4 = self.model_M[3](x_m3)
        return x_m4
    ###采用辅助分支
    def forward(self, x):
        a,b = self.low_feature(x)
        x1 = self.model1(x)
        x_d = self.model_D(x1)
        x_m = self.low_mid(x_d,a,b)
        x_u = self.model_U(x_m)
        x2 = self.model2(x_u)
        return x2
    
    ##不采用辅助分支---用于对比4*Local 和 4*Global
    # def forward(self, x):
    #     x1 = self.model1(x)
    #     x_d = self.model_D(x1)
    #     x_m = self.model_M(x_d)
    #     x_u = self.model_U(x_m)
    #     x2 = self.model2(x_u)
    #     return x2
    
    #######采用低级特征指导过渡层 ---edit 2022/5/22 利用分支2产生k,v x_d和x_m2产生q 通过qkv计算权重系数m指导x_d和x_m2,同时加上v，m相当于a，v相当于b######
    ##效果不如low_mid
    # def low_mid_v2(self,x, k, v):
    #     B,C,H,W = x.shape #1,256,64,64
    #     ##kv
    #     k = k.permute(0,2,3,1).view(B, -1, C)
    #     v_r = v.permute(0,2,3,1).view(B, -1,C)
    #     ##q1
    #     q1 = self.q1(x)
    #     q1 = q1.permute(0,2,3,1).view(B, -1, C)
    #     ##第一组系数m1，v
    #     cross_attn1 = self.cross_attn(q1, k, v_r)[0]
    #     m1 = cross_attn1.view(B, H, W, C).permute(0,3,1,2)  ##用于指导的系数m（相当于a)
    #     x_d = m1*x+v  ###优化后的x_d, v用于补充低级特征信息相当于之前的b
    #     x_m1 = self.model_M[0](x_d)
    #     x_m2 = self.model_M[1](x_m1)
    #     ##q2
    #     q2 = self.q2(x_m2)
    #     q2 = q2.permute(0,2,3,1).view(B, -1, C)
    #     ##第二组系数m2,v
    #     cross_attn2 = self.cross_attn(q2, k, v_r)[0]
    #     m2 = cross_attn2.view(B,H,W,C).permute(0,3,1,2)
    #     x_m2_n = m2*x_m2+v  ###优化后的x_m2
    #     x_m3 = self.model_M[2](x_m2_n)
    #     x_m4 = self.model_M[3](x_m3)
    #     return x_m4
    # def forward(self, x):
    #     k,v = self.low_feature(x)
    #     x1 = self.model1(x)
    #     x_d = self.model_D(x1)
    #     x_m = self.low_mid_v2(x_d,k,v)
    #     x_u = self.model_U(x_m)
    #     x2 = self.model2(x_u)
    #     return x2
 
 
 
class CSG(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=2, n_blocks=4, num_heads=16, 
                 window_size=16, scales=4):
        super().__init__()
        activation = nn.ReLU(True)
        norm_layer1 = nn.BatchNorm2d    
        self.model1 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, 
                                    padding=0), norm_layer1(ngf), activation, PoolFormerBlock(ngf))
        ##down_sampling
        model_d = []
        for i in range(n_downsampling+1):
            mult = 2**i
            model_d += [DS(ngf*mult, ngf*mult*2, norm_layer1(mult*ngf*2), activation)]
        self.model_D = nn.Sequential(*model_d)
        model_m = []
        mult = 2**(n_downsampling+1)
        ###2*SEC+2*CES
        for i in range(n_blocks):
            if i % 2 == 0:
                model_m += [SEC(mult*ngf, activation=activation, num_heads=num_heads, window_size=window_size)]
            else:
                model_m += [CES(mult*ngf, activation=activation, num_heads=num_heads, window_size=window_size)]
        self.model_M = nn.Sequential(*model_m)
        ##up_sampling
        model_u = []
        for i in range(n_downsampling+1):
            mult = 2**(n_downsampling+1-i)
            model_u += [US(ngf*mult, int(ngf*mult/2), norm_layer1(int(ngf*mult/2)), activation)]
        self.model_U = nn.Sequential(*model_u)
        self.model2 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                    nn.Tanh())
        ###IGC
        self.low_feature = Low_feature(input_nc, init_nc=ngf, dim=ngf*8, num_sf=3, num_heads=num_heads)
    def middle(self, x_downsample):
        x0 = x_downsample[0]  #(1,64,256,256)
        x1 = x_downsample[1]  #(1,128,128,128)
        x2 = x_downsample[2]  #(1,256,64,64)
        x_middle = []   ###[x1_m, x0_m2, x0_m1]
        for inx, layer_up in enumerate(self.model_U):
            x2 = layer_up(x2) # inx=0 (1,128,128,128)=x1_m   inx=1 (1,64,256,256)=x0_m2  
            x_middle.append(x2)
            if inx == 1:
                x0_m1 = layer_up(x1)
                x_middle.append(x0_m1)  #inx=1 (1,64,256,256)=x0_m1
        return x_middle
    ###ax+b### ###验证1.ax 2.b####
    def low_mid(self,x,a,b):
        x_d = a*x + b
        #x_d = a*x   ###x_d = a*x+b(完整) 验证1.ax 2.x+b
        #x_d = x + b
        x_m1 = self.model_M[0](x_d)
        #x_m1 = a*x_m1+b 
        x_m2 = self.model_M[1](x_m1)
        x_m2 = a*x_m2 + b
        #x_m2 = a*x_m2
        #x_m2 = x_m2 + b
        x_m3 = self.model_M[2](x_m2)
        #x_m3 = a*x_m3+b 
        x_m4 = self.model_M[3](x_m3)
        return x_m4
    def forward(self, x):
        a,b = self.low_feature(x)
        x1 = self.model1(x)
        x_d = self.model_D(x1)
        x_m = self.low_mid(x_d,a,b)
        x_u = self.model_U(x_m)
        x2 = self.model2(x_u)
        return x2
 
 
 
        
        
####定义中间过度块全是Swin的网络---用于对比---edit 2022/5/30    
##定义单个中间过度模块包含两个SwinTransformer（wo-window shift + w-window shift）
class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size):
        super().__init__()
        swin_block = []
        for i in range(2):
            swin_block +=[SwinTransformerBlock(dim, input_resolution=(64,64),
                         num_heads=num_heads, window_size=window_size,
                                 shift_size=0 if (i % 2 == 0) else window_size // 2,
                                 mlp_ratio=4.0,
                                 qkv_bias=True, qk_scale=None,
                                 drop=0.0, attn_drop=0.0,
                                 drop_path=0.2,
                                 norm_layer=nn.LayerNorm)]
        self.Swin_Block = nn.Sequential(*swin_block)
    def forward(self, x):
        x = self.Swin_Block(x)
        return x    
class SwinG(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=2, n_blocks=9, num_heads=16, window_size=16):
        super().__init__()
        activation = nn.ReLU(True)
        norm_layer1 = nn.BatchNorm2d    
        ##1*64*256*256 model1和model_D没有poolformer
        self.model1 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0), norm_layer1(ngf), activation)
        ##down_sampling  n_downsampling+1:512  n_downsampling:256
        model_d = []
        for i in range(n_downsampling):
            mult = 2**i
            model_d += [DS(ngf*mult, ngf*mult*2, norm_layer1(mult*ngf*2), activation)]
        model_d += [DS(ngf*mult*2, ngf*mult*2, norm_layer1(mult*ngf*2), activation)]
        self.model_D = nn.Sequential(*model_d)
        ##中间过渡层，采用ResSAR模块
        model_m = []
        mult = 2**(n_downsampling)
        for i in range(n_blocks):
            model_m += [SwinBlock(ngf*mult, num_heads, window_size)]
        self.model_M = nn.Sequential(*model_m)
        #print(self.model_M.state_dict().keys())
        ##up_sampling
        model_u = []
        for i in range(n_downsampling):
            mult = 2**(n_downsampling-i)
            model_u += [US(ngf*mult, int(ngf*mult/2), norm_layer1(int(ngf*mult/2)), activation)]
        model_u += [US(int(ngf*mult/2), int(ngf*mult/2), norm_layer1(int(ngf*mult/2)), activation)]
        self.model_U = nn.Sequential(*model_u)
        self.model2 = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh())        
    
    def forward(self, x):
        x1 = self.model1(x)
        x_d = self.model_D(x1)
        B,C,H,W = x_d.shape
        x_d_r = x_d.permute(0,2,3,1).view(B,-1,C)  ##(B, H*W, C)
        x_m = self.model_M(x_d_r)
        x_m_r = x_m.view(B,H,W,C).permute(0,3,1,2) #(B,C,H,W)
        x_u = self.model_U(x_m_r)
        x2 = self.model2(x_u)
        return x2
        
         
        
        
        
        
    


###############-对SAR图像进行去噪处理（尝试采用卷积实现）--edit2021/11/22-#####################################
##定义对SAR的去噪模块
class DeNoise(nn.Module):
    def __init__(self, in_dim, out_dim, norm=nn.BatchNorm2d, activation1=nn.Tanh()):
        super().__init__()
        self.conv111 = nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, stride=1)
        self.conv331 = nn.Conv2d(in_dim, out_dim, kernel_size=3, padding=1, stride=1)
        self.norm1 = norm(out_dim)
        self.activation1 = activation1
        self.conv112 = nn.Conv2d(out_dim, in_dim, kernel_size=1, padding=0, stride=1)
        self.conv332 = nn.Conv2d(out_dim, in_dim, kernel_size=3, padding=1, stride=1)
        self.norm2 = norm(in_dim)
        print(in_dim)
        print(out_dim)
    def forward(self,x):
        ##第一个残差块
        #分支1经过1*1conv
        x1 = self.conv111(x)
        #分支2经过3*3conv
        x2 = self.conv331(x)
        #两分支合并
        x_1 = x1 + x2
        x_1 = self.norm1(x_1)
        x_1 = self.activation1(x_1)
        ##第二个残差块
        x3 = self.conv112(x_1)
        x4 = self.conv332(x_1)
        x_2 = x3 + x4
        x_2 = self.norm2(x_2)
        x_2 = self.activation1(x_2)
        x = x + x_2
        ##最后用Tanh激活函数
        x = self.activation1(x)
        return x     
    
    
    
    
##########原代码#####################################################################################   
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
        self.num_D = num_D  ##3
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat: ##true                               
                for j in range(n_layers+2):
                    setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD, 'model'+str(j)))                                   
            else:
                setattr(self, 'layer'+str(i), netD.model)   ##setattr:给object赋值

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))   
            return result[1:]   # 第一个判别器：result = [(1,6,256,256) (1,64,129,129) (1,128,65,65) (1,256,33,33) (1,512,34,34) (1,1,35,35)]
                                # 第二个判别器：result = [(1,6,128,128) (1,64,65,65) (1,128,33,33) (1,256,17,17) (1,512,18,18) (1,1,19,19)]
                                # 第三个判别器：result = [(1,6,64,64) (1,64,33,33) (1,128,17,17) (1,256,9,9) (1,512,10,10) (1,1,11,11)]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D ##3
        result = []
        input_downsampled = input # (1,6,256,256)
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(num_D-1-i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(num_D-1-i))  ##getattr:获取object的属性或值
            result.append(self.singleD_forward(model, input_downsampled))#i=0 第一个判别器+输入(1,6,256,256) i=1 第二个判别器+输入(1,6,128,128)
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
        padw = int(np.ceil((kw-1.0)/2)) #2
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


#####定义采用Transformer的多尺度判别器---edit 2022/6/2
###定义Transformer### ---edit 2022/6/2 
class SpaceFormer_D(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4,drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, cpe_act=False):
        super().__init__()
        self.cpe = nn.ModuleList([CPosEnc(dim=dim,k=3,act = cpe_act),
                                  CPosEnc(dim=dim,k=3,act = cpe_act)])
        self.attn = nn.MultiheadAttention(dim, num_heads,dropout=0)
        self.norm = norm_layer(dim)
        self.mlp = MlpC(in_features=dim, hidden_features=dim*mlp_ratio, act_layer=act_layer,drop=drop_path)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
    def forward(self, x):
        B,C,H,W = x.shape
        x = x.permute(0,2,3,1).view(B,-1,C)
        x = self.cpe[0](x)
        cur = self.attn(x,x,x)[0]
        cur = self.norm(cur)
        x = x + self.drop_path(cur)
        x = self.cpe[1](x)
        x = x + self.drop_path(self.norm(self.mlp(x)))
        x = x.view(B,H,W,C).permute(0,3,1,2)
        return x
class NlayerD_wSF(nn.Module):
    def __init__(self,  input_nc, ndf=64, n_layers=3,  norm_layer=nn.BatchNorm2d, getIntermFeat=False, num_heads=16):
        super().__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=3, stride=1, padding=1),
            SpaceFormer_D(ndf, num_heads=num_heads, mlp_ratio=4, drop_path=0)]] #(64,256,256)
        nf = ndf #64
        for n in range(1, n_layers): ##1,2  (128,128,128)-->(256,64,64)  
            nf_prev = nf
            nf = min(nf * 2, 512) #128,256
            sequence += [[
                nn.Conv2d(nf_prev, nf, kernel_size=3, stride=2, padding=1),
                SpaceFormer_D(nf, num_heads=num_heads, mlp_ratio=4, drop_path=0)]]
            
        nf_prev = nf
        nf = min(nf * 2, 512)
        sequence += [[
            nn.Conv2d(nf_prev, nf, kernel_size=3, stride=2, padding=1),
            norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]   ## (512,32,32)  
        sequence += [[nn.Conv2d(nf, 1, kernel_size=3, stride=1, padding=1)]]  ##(1,32,32)  
        
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
            for n in range(self.n_layers+2):
                model = getattr(self, 'model'+str(n))
                res.append(model(res[-1]))
            return res[1:]
        else:
            return self.model(input)  
class MultiD_wSF(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, 
                 use_sigmoid=False, num_D=3, getIntermFeat=False, num_heads=16):
        super(MultiD_wSF, self).__init__()
        self.num_D = num_D  ##3
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat
     
        for i in range(num_D):
            netD_L = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            netD_G = NlayerD_wSF(input_nc, ndf, n_layers, norm_layer, getIntermFeat, num_heads=num_heads)
            if getIntermFeat: ##true
                if i == 0:                               
                    for j in range(n_layers+2):
                        setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD_G, 'model'+str(j))) 
                else:
                    for j in range(n_layers+2):
                        setattr(self, 'scale'+str(i)+'_layer'+str(j), getattr(netD_L, 'model'+str(j)))                                 
            else:
                if i == 0:
                    setattr(self, 'layer'+str(i), netD_L.model)   ##setattr:给object赋值
                else:
                    setattr(self, 'layer'+str(i), netD_G.model)
        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)

    def singleD_forward(self, model, input):
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result.append(model[i](result[-1]))   
            return result[1:]   # 第一个判别器：result = [(1,6,256,256) (1,64,256,256) (1,128,128,128) (1,256,64,64) (1,512,32,32) (1,1,32,32)]
                                # 第二个判别器：result = [(1,6,128,128) (1,64,65,65) (1,128,33,33) (1,256,17,17) (1,512,18,18) (1,1,19,19)]
                                # 第三个判别器：result = [(1,6,64,64) (1,64,33,33) (1,128,17,17) (1,256,9,9) (1,512,10,10) (1,1,11,11)]
        else:
            return [model(input)]

    def forward(self, input):        
        num_D = self.num_D ##3
        result = []
        input_downsampled = input # (1,6,256,256)
        for i in range(num_D):
            if self.getIntermFeat:
                model = [getattr(self, 'scale'+str(i)+'_layer'+str(j)) for j in range(self.n_layers+2)]
            else:
                model = getattr(self, 'layer'+str(i))  ##getattr:获取object的属性或值
            result.append(self.singleD_forward(model, input_downsampled))#i=0 第一个判别器+输入(1,6,256,256) i=1 第二个判别器+输入(1,6,128,128)
            if i != (num_D-1):
                input_downsampled = self.downsample(input_downsampled)
        return result       

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


