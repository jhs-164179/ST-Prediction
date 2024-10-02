# InvVP
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

from module import Inv2d, Inv3d

# Patch Embedding
class PatchEmbed(nn.Module):
    def __init__(self, input_dim, embed_dim, patch_size, upsampling=False):
        super(PatchEmbed, self).__init__()
        if upsampling:
            self.proj = nn.ConvTranspose2d(input_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        else:
            self.proj = nn.Conv2d(input_dim, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.BatchNorm2d(embed_dim)
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.upsampling = upsampling
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.contiguous().view(B*T, C, H, W)
        # x = x.reshape(B*T, C, H, W)
        x = self.proj(x)
        x = self.norm(x)
        if self.upsampling:
            x = x.view(B, T, self.embed_dim, H*self.patch_size, W*self.patch_size)
        else:
            x = x.view(B, T, self.embed_dim, H//self.patch_size, W//self.patch_size)
        return x
    

class LayerNorm(nn.Module):
    r""" From ConvNeXt (https://arxiv.org/pdf/2201.03545.pdf)
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
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


# Conv2Former-based
class ConvMod(nn.Module):
    def __init__(self, dim, kernel_size, drop_path=.1, init_value=1e-2, inv=True):
        super().__init__()
        self.norm1 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        self.norm2 = LayerNorm(dim, eps=1e-6, data_format="channels_first")
        if inv:
            padding = (kernel_size-1)//2
            self.a = nn.Sequential(
                nn.Conv2d(dim, dim, 1),
                nn.GELU(),
                Inv2d(
                    in_channels=dim, out_channels=dim, kernel_size=kernel_size, padding=padding, groups=dim
                )
            )
            # self.v = Inv2d(in_channels=dim, out_channels=dim, kernel_size=1)
            # self.proj = nn.Conv2d(dim, dim, 1)
        else:
            self.a = nn.Sequential(
                nn.Conv2d(dim, dim, 1),
                nn.GELU(),
                nn.Conv2d(dim, dim, kernel_size=kernel_size, padding='same', groups=dim)
            )
            # self.v = nn.Conv2d(dim, dim, 1)
            # self.proj = nn.Conv2d(dim, dim, 1)
        self.v = nn.Conv2d(dim, dim, 1)
        self.proj = nn.Conv2d(dim, dim, 1)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2'}

    def forward(self, x):
        # print(x.shape)
        # print(self.norm1(x).shape)
        # print(self.a(self.norm1(x).shape))
        # print(self.layer_scale_1.shape)
        # print('hello')
        # print(x.shape)
        a = self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.a(self.norm1(x)))
        # print('hello')
        x = self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.v(self.norm2(x)))
        x = a * x
        x = self.proj(x)

        return x


class EncoderLayer_S(nn.Module):
    def __init__(self, dim, kernel_size=7, drop_path=.1, init_value=1e-2):
        super(EncoderLayer_S, self).__init__()
        self.attn = ConvMod(dim, kernel_size=kernel_size, drop_path=drop_path, init_value=init_value)
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        B, T, C, H, W = x.shape
        # print(x.shape)
        x = x.view(-1, C, H, W)
        # print(x.shape)
        a = self.attn(x)
        # print(a.shape)
        x = x + a
        x = self.norm(x)
        x = x.view(B, T, C, H, W)
        return x, a


# LSB-based
class FeedForward(nn.Module): 
    def __init__(self, dim, dilation=2, ratio=2, drop_path=.1, init_value=1e-2, inv=True):
        super().__init__()
        # self.ff1 = nn.Sequential(
        #     nn.Conv3d(dim, ratio * dim, 3, 1, 1, bias=False),
        #     nn.GroupNorm(1, ratio * dim),
        #     nn.SiLU(inplace=True),
        # )
        # self.ff2 = nn.Sequential(
        #     nn.Conv3d(ratio * dim, dim, 3, 1, 1, bias=False),
        #     nn.GroupNorm(1, dim),
        #     nn.SiLU(inplace=True),
        # )
        if inv:
            # 3 : kernel_size
            padding = ((3-1) * dilation) // 2
            # self.ff1_1 = Inv3d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False, groups=dim)
            # self.ff1_2 = Inv3d(in_channels=dim, out_channels=dim, kernel_size=3, bias=False, padding=(1, 1, 1), groups=dim, dilation=(1, 1, 1))
            # self.ff1_act = nn.GELU()
            self.ff1 = nn.Sequential(
                Inv3d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, padding=1, bias=False, groups=dim),
                #     nn.GroupNorm(1, dim),
                # padding for Inv3d
                # Inv3d(in_channels=dim, out_channels=dim, kernel_size=3, bias=False, padding=(1, 1, 1), groups=dim, dilation=(dilation, 1, 1)),
                Inv3d(in_channels=dim, out_channels=dim, kernel_size=3, bias=False, padding=(1, 1, 1), groups=dim, dilation=(1, 1, 1)),
                nn.GELU(),
                # nn.GroupNorm(1, ratio * dim),
                # nn.SiLU(inplace=True),

                # nn.GELU(),
            )
        else:
            self.ff1 = nn.Sequential(
                nn.Conv3d(dim, dim, 3, 1, 1, bias=False, groups=dim),
                #     nn.GroupNorm(1, dim),
                nn.Conv3d(dim, dim, kernel_size=3, bias=False, padding='same', groups=dim, dilation=(dilation, 1, 1)),
                nn.GELU(),
                # nn.GroupNorm(1, ratio * dim),
                # nn.SiLU(inplace=True),

                # nn.GELU(),
            )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

        self.layer_scale_1 = nn.Parameter(init_value * torch.ones((dim)), requires_grad=True)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv3d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'layer_scale_1', 'layer_scale_2'}
    
    def forward(self, x):
        # print(x.permute(0, 2, 1, 3, 4).shape)
        # print(self.ff1(x.permute(0, 2, 1, 3, 4)).shape)
        # print(self.layer_scale_1.shape)
        # print(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).shape)

        # print('Start FFN')
        # x = x.permute(0, 2, 1, 3, 4)
        # print(x.shape)

        # print('FFN stage1')
        # x = self.ff1_1(x)
        # print(x.shape)

        # print('FFN stage2')
        # x = self.ff1_2(x)
        # print(x.shape)

        # print('FFN Activation')
        # x = self.ff1_act(x)
        # print(x.shape)

        x = self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) * self.ff1(x.permute(0, 2, 1, 3, 4)))
        x = x.permute(0, 2, 1, 3, 4)

        # x = self.ff2(self.ff1(x.permute(0, 2, 1, 3, 4))).permute(0, 2, 1, 3, 4)
        # x = self.ff1(x.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        return x


class EncoderLayer_T(nn.Module):
    def __init__(self, dim, ratio=2, kernel_size=7, dilation=2, drop_path=.1, init_value=1e-2): # dim --> seq_len * dim
        super(EncoderLayer_T, self).__init__()
        # self.attn = STBlock(dim, ratio=ratio, kernel_size=kernel_size)
        self.attn = FeedForward(dim, ratio=ratio, drop_path=drop_path, init_value=init_value, dilation=dilation)
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        B, T, C, H, W = x.shape
        a = self.attn(x)
        x = x + a
        x = self.norm(x.contiguous().view(-1, C, H, W))
        x = x.view(B, T, C, H, W)
        return x, a


class DecoderLayer_S(nn.Module):
    def __init__(self, dim, ratio=2, mode='normal'):
        super(DecoderLayer_S, self).__init__()
        if mode == 'normal':
            self.mlp = nn.Sequential(
                nn.Conv2d(dim, dim*ratio, 1),

                nn.GELU(),
                nn.Conv2d(dim*ratio, dim, 1),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Conv2d(dim, dim, 1),
                nn.GELU()
            )
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x, enc_attn):
        B, T, C, H, W = x.shape
        x = x.contiguous().view(-1, C, H, W)
        # a = enc_attn
        x = x + enc_attn
        x = self.mlp(x)
        x = self.norm(x)
        x = x.contiguous().view(B, T, C, H, W)
        return x


class DecoderLayer_T(nn.Module):
    def __init__(self, dim, ratio=2, mode='normal'): # dim --> seq_len * dim
        super(DecoderLayer_T, self).__init__()
        if mode == 'normal':
            self.mlp = nn.Sequential(
                nn.Conv3d(dim, dim*ratio, 3, 1, 1),
                nn.GELU(),
                nn.Conv3d(dim*ratio, dim, 3, 1, 1),
            )
        else:
            self.mlp = nn.Sequential(
                nn.Conv3d(dim, dim, 1),
                nn.GELU()
            )
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x, enc_attn):
        B, T, C, H, W = x.shape
        x = x + enc_attn
        x = self.mlp(x.permute(0, 2, 1, 3, 4)).permute(0, 2, 1, 3, 4)
        x = self.norm(x.contiguous().view(-1, C, H, W))
        x = x.view(B, T, C, H, W)
        return x


class InvVP_Model(nn.Module):
    # def __init__(self, input_shape, hidden_dim, N, NE, ND, patch_size=2, seq_len=10, **kwargs):
    def __init__(self, in_shape, hidden_dim, N, NE, ND, patch_size=2, kernel_size=7, ratio=2, drop_path=.1, init_value=1e-2, seq_len=10, mode='normal', **kwargs):
        super(InvVP_Model, self).__init__()
        # N must be even(N=number of each layers(embed, enc, dec))
        # T, C, H, W = input_shape
        T, C, H, W = in_shape
        embed = []
        layers = []
        hidden_dims = []
        for i in range(N//2):
            hidden_dims.append(hidden_dim*(2**i))
        hidden_dims = hidden_dims + hidden_dims[::-1]
        
        for i in range(N):
            if i == 0:
                embed.append(PatchEmbed(C, hidden_dims[i], patch_size=patch_size))
            elif i < N//2:
                embed.append(PatchEmbed(hidden_dims[i-1], hidden_dims[i], patch_size=patch_size))
            else:
                embed.append(PatchEmbed(hidden_dims[i-1], hidden_dims[i], patch_size=patch_size, upsampling=True))
        
        for i in range(N):
            if i < N//2:
                enc = []
                for _ in range(NE):
                    # enc.append(ES(dim=hidden_dims[i]))
                    # enc.append(ET(dim=hidden_dims[i]))
                    enc.append(EncoderLayer_T(dim=hidden_dims[i], kernel_size=kernel_size, drop_path=drop_path, init_value=init_value))
                    enc.append(EncoderLayer_S(dim=hidden_dims[i], kernel_size=kernel_size, drop_path=drop_path, init_value=init_value))
                layers.append(nn.ModuleList(enc))
            else:
                dec = []
                for _ in range(ND):
                    # dec.append(DS(dim=hidden_dims[i]))
                    # dec.append(DT(dim=hidden_dims[i]))
                    dec.append(DecoderLayer_T(dim=hidden_dims[i], ratio=ratio, mode=mode))
                    dec.append(DecoderLayer_S(dim=hidden_dims[i], ratio=ratio, mode=mode))
                layers.append(nn.ModuleList(dec))

        self.embed = nn.ModuleList(embed)
        self.layers = nn.ModuleList(layers)
        self.finalembed = PatchEmbed(hidden_dim, hidden_dim, patch_size=patch_size, upsampling=True)
        self.final = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], C, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()
        )
        self.N = N
        self.NE = NE
        self.ND = ND

    def forward(self, x):
        B, T, C, H, W = x.shape
        enc_attns = []
        idx = 0
        for i in range(self.N):
            if i != self.N//2:
                x = self.embed[i](x)
            if i < self.N//2:
                for j in range(self.NE):
                    x, s_att = self.layers[i][j*2](x)
                    x, t_att = self.layers[i][j*2+1](x)
                enc_attns.append(t_att.clone())
                enc_attns.append(s_att.clone())
            else:
                for j in range(self.ND):
                    s_att = enc_attns[::-1][idx]
                    t_att = enc_attns[::-1][idx+1]
                    x = self.layers[i][j*2](x, s_att)
                    x = self.layers[i][j*2+1](x, t_att)
                idx += 2
        x = self.finalembed(x)
        b, t, c, h, w = x.shape
        x = x.view(-1, c, h, w)
        x = self.final(x)
        x = x.view(B, T, C, H, W)
        return x