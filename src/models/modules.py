from collections import OrderedDict
import torch
from torch import nn
import torch.nn.functional as F
from monai.utils.module import look_up_option
from monai.utils.enums import SkipMode
import math


class TabularModule(nn.Module):
    def __init__(self, tab_dim=6,
                 channel_dim=2,
                 frame_dim=None,
                 hw_size=None,
                 bottleneck_dim = None,
                 module=None):
        super(TabularModule, self).__init__()
        self.channel_dim = channel_dim
        self.tab_dim = tab_dim
        self.frame_dim = frame_dim
        self.hw_size = hw_size
        self.bottleneck_dim = bottleneck_dim


# DAFT based on: https://github.com/ai-med/DAFT/blob/master/daft/networks/vol_blocks.py
class DAFT(TabularModule):
    def __init__(self,
                 bottleneck_dim=7,
                 **kwargs
                 ):
        super(DAFT, self).__init__(**kwargs)
        self.bottleneck_dim = bottleneck_dim
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        layers = [
            ("aux_base", nn.Linear(self.tab_dim + self.channel_dim, self.bottleneck_dim, bias=False)),
            ("aux_relu", nn.ReLU()),
            ("aux_out", nn.Linear(self.bottleneck_dim, 2 * self.channel_dim, bias=False)),
        ]
        self.aux = nn.Sequential(OrderedDict(layers))

    def forward(self, feature_map, x_aux):
        x_aux = x_aux.squeeze(dim=1)
        squeeze = self.global_pool(feature_map)
        squeeze = squeeze.view(squeeze.size(0), -1)
        squeeze = torch.cat((squeeze, x_aux), dim=1)

        attention = self.aux(squeeze)
        v_scale, v_shift = torch.split(attention, self.channel_dim, dim=1)
        l2 = torch.norm(v_scale, p=2) + torch.norm(v_shift, p=2)
        v_scale = v_scale.view(*v_scale.size(), 1, 1, 1).expand_as(feature_map)
        v_shift = v_shift.view(*v_shift.size(), 1, 1, 1).expand_as(feature_map)
        out = ((1-v_scale) * feature_map) + v_shift

        return out, l2, ((v_scale, v_shift), None)


# FiLM based on: https://github.com/ai-med/DAFT/blob/master/daft/networks/vol_blocks.py
class FiLM(TabularModule):
    def __init__(self,
                 bottleneck_dim=7,
                 **kwargs
                 ):
        super(FiLM, self).__init__(**kwargs)
        self.bottleneck_dim = bottleneck_dim
        # if self.hw_size is None or self.frame_dim is None:
        #     self.global_pool = nn.AdaptiveAvgPool3d(output_size=1)
        # else:
        #     if not isinstance(self.hw_size, tuple):
        #         self.hw_size = tuple(self.hw_size)
        #     self.global_pool = nn.AvgPool3d(kernel_size=((self.hw_size) + (self.frame_dim,)))
        self.dim=1
        
        layers = [
            ("aux_base", nn.Linear(self.tab_dim, self.bottleneck_dim, bias=False)),
            ("aux_relu", nn.ReLU()),
            ("aux_out", nn.Linear(self.bottleneck_dim, 2 * self.channel_dim, bias=False)),
        ]
        self.aux = nn.Sequential(OrderedDict(layers))

    def forward(self, feature_map, x_aux):
        attention = self.aux(x_aux)
        if len(x_aux.shape) > 2:
            self.dim = 2
        v_scale, v_shift = torch.split(attention, self.channel_dim, dim=self.dim)
        l2 = torch.norm(v_scale, p=2) + torch.norm(v_shift, p=2)
        # print(f"\tConditional module: f_map at scale:{feature_map.shape[2:]}, mean:{v_scale.mean().item():.4f}, std:{v_scale.std().item():.4f} max:{v_scale.max().item():.4f}, min: {v_scale.min().item():.4f}, shift mean: {v_shift.mean().item():.4f}.")
        v_scale = v_scale.view(*v_scale.size(), 1, 1, 1).expand_as(feature_map)
        v_shift = v_shift.view(*v_shift.size(), 1, 1, 1).expand_as(feature_map)
        out = ((1-v_scale) * feature_map) + v_shift
        return out, l2, ((v_scale, v_shift), None)


# _______________ TabAttention ____________________________________________________________________
# _______________ CBAM based on: https://github.com/Jongchan/attention-module/blob/master/MODELS/cbam.py ___________________

class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class ChannelGate(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], tabattention=False, n_tab=6,
                 tabular_branch=False):
        super(ChannelGate, self).__init__()
        self.tabattention = tabattention
        self.n_tab = n_tab
        self.gate_channels = gate_channels
        self.tabular_branch = tabular_branch
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
        )
        self.pool_types = pool_types
        if tabattention:
            self.pool_types = ['avg', 'max', 'tab']
            if self.tabular_branch:
                self.tab_embedding = nn.Identity()
            else:
                self.tab_embedding = nn.Sequential(
                    nn.Linear(n_tab, gate_channels // reduction_ratio),
                    nn.ReLU(),
                    nn.Linear(gate_channels // reduction_ratio, gate_channels)
                )

    def forward(self, x, tab=None):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type == 'avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(avg_pool)
            elif pool_type == 'max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(max_pool)
            elif pool_type == 'lp':
                lp_pool = F.lp_pool2d(x, 2, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))
                channel_att_raw = self.mlp(lp_pool)
            elif pool_type == 'lse':
                # LSE pool only
                lse_pool = logsumexp_2d(x)
                channel_att_raw = self.mlp(lse_pool)
            elif pool_type == 'tab':
                embedded = self.tab_embedding(tab)
                embedded = torch.reshape(embedded, (-1, self.gate_channels))
                pool = self.mlp(embedded)
                channel_att_raw = pool

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale


class TemporalMHSA(nn.Module):
    def __init__(self, input_dim=2, seq_len=16, heads=2):
        super(TemporalMHSA, self).__init__()

        self.input_dim = input_dim
        self.seq_len = seq_len
        self.embedding_dim = 4
        self.head_dim = self.embedding_dim // heads
        self.heads = heads
        self.qkv = nn.Linear(self.input_dim, self.embedding_dim * 3)
        self.rel = nn.Parameter(torch.randn([1, 1, seq_len, 1]), requires_grad=True)
        self.o_proj = nn.Linear(self.embedding_dim, 1)

    def forward(self, x):
        batch_size, seq_length, _ = x.size()
        qkv = self.qkv(x)
        qkv = qkv.reshape(batch_size, seq_length, self.heads, 3 * self.head_dim)
        qkv = qkv.permute(0, 2, 1, 3)  # [Batch, Head, SeqLen, Dims]
        q, k, v = qkv.chunk(3, dim=-1)

        d_k = q.size()[-1]
        k = k + self.rel.expand_as(k)
        attn_logits = torch.matmul(q, k.transpose(-2, -1))
        attn_logits = attn_logits / math.sqrt(d_k)
        attention = F.softmax(attn_logits, dim=-1)
        values = torch.matmul(attention, v)
        values = values.permute(0, 2, 1, 3)  # [Batch, SeqLen, Head, Dims]
        values = values.reshape(batch_size, seq_length, self.embedding_dim)  # [Batch, SeqLen, EmbeddingDim]
        x_out = self.o_proj(values)

        return x_out


class TemporalGate(nn.Module):
    def __init__(self, gate_frames, reduction_ratio=16, pool_types=['avg', 'max'], tabattention=False, n_tab=6,
                 temporal_mhsa=False, tabular_branch=False):
        super(TemporalGate, self).__init__()
        self.tabattention = tabattention
        self.tabular_branch = tabular_branch
        self.n_tab = n_tab
        self.gate_frames = gate_frames
        self.temporal_mhsa = temporal_mhsa
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_frames, gate_frames // 2),
            nn.ReLU(),
            nn.Linear(gate_frames // 2, gate_frames)
        )
        self.pool_types = pool_types
        if tabattention:
            self.pool_types = ['avg', 'max', 'tab']
            if self.tabular_branch:
                self.tab_embedding = nn.Sequential(nn.Linear(n_tab, gate_frames), nn.ReLU())
            else:
                self.tab_embedding = nn.Sequential(
                    nn.Linear(n_tab, gate_frames // 2),
                    nn.ReLU(),
                    nn.Linear(gate_frames // 2, gate_frames)
                )
        if self.temporal_mhsa:
            if tabattention:
                self.mhsa = TemporalMHSA(input_dim=3, seq_len=self.gate_frames)
            else:
                self.mhsa = TemporalMHSA(input_dim=2, seq_len=self.gate_frames)

    def forward(self, x, tab=None):
        if not self.temporal_mhsa:
            channel_att_sum = None
            for pool_type in self.pool_types:
                if pool_type == 'avg':
                    avg_pool = F.avg_pool3d(x, (x.size(2), x.size(3), x.size(4)),
                                            stride=(x.size(2), x.size(3), x.size(4)))
                    channel_att_raw = self.mlp(avg_pool)
                elif pool_type == 'max':
                    max_pool = F.max_pool3d(x, (x.size(2), x.size(3), x.size(4)),
                                            stride=(x.size(2), x.size(3), x.size(4)))
                    channel_att_raw = self.mlp(max_pool)
                elif pool_type == 'tab':
                    embedded = self.tab_embedding(tab)
                    embedded = torch.reshape(embedded, (-1, self.gate_frames))
                    pool = self.mlp(embedded)
                    channel_att_raw = pool

                if channel_att_sum is None:
                    channel_att_sum = channel_att_raw
                else:
                    channel_att_sum = channel_att_sum + channel_att_raw

            scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand_as(x)
            return x * scale
        else:
            avg_pool = F.avg_pool3d(x, (x.size(2), x.size(3), x.size(4))).reshape(-1, self.gate_frames, 1)
            max_pool = F.max_pool3d(x, (x.size(2), x.size(3), x.size(4))).reshape(-1, self.gate_frames, 1)

            if self.tabattention:
                embedded = self.tab_embedding(tab)
                tab_embedded = torch.reshape(embedded, (-1, self.gate_frames, 1))
                concatenated = torch.cat((avg_pool, max_pool, tab_embedded), dim=2)
            else:
                concatenated = torch.cat((avg_pool, max_pool), dim=2)

            scale = torch.sigmoid(self.mhsa(concatenated)).unsqueeze(2).unsqueeze(3).expand_as(x)

            return x * scale


def logsumexp_2d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class SpatialGate(nn.Module):
    def __init__(self, tabattention=False, n_tab=6, input_size=(8, 8), tabular_branch=False):
        super(SpatialGate, self).__init__()
        self.tabattention = tabattention
        self.tabular_branch = tabular_branch
        self.n_tab = n_tab
        self.input_size = input_size
        kernel_size = 7
        self.compress = ChannelPool()
        in_planes = 3 if tabattention else 2
        self.spatial = BasicConv(in_planes, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)
        if self.tabattention:
            if self.tabular_branch:
                self.tab_embedding = nn.Sequential(nn.Linear(n_tab, input_size[0] * input_size[1]), nn.ReLU())
            else:
                self.tab_embedding = nn.Sequential(
                    nn.Linear(n_tab, input_size[0] * input_size[1] // 2),
                    nn.ReLU(),
                    nn.Linear(input_size[0] * input_size[1] // 2, input_size[0] * input_size[1])
                )

    def forward(self, x, tab=None):
        x_compress = self.compress(x)
        if self.tabattention:
            embedded = self.tab_embedding(tab)
            embedded = torch.reshape(embedded, (-1, 1, self.input_size[0], self.input_size[1]))
            x_compress = torch.cat((x_compress, embedded), dim=1)

        x_out = self.spatial(x_compress)
        scale = torch.sigmoid(x_out)  # broadcasting
        return x * scale


class TabAttention(TabularModule):
    def __init__(self, channel_dim, frame_dim, reduction_ratio=16, pool_types=['avg', 'max'], tabattention=True,
                 tab_dim=6, temporal_mhsa=True, hw_size=(8, 8), temporal_attention=True, tabular_branch=False,
                 cam_sam=True, **kwargs):
        super(TabAttention, self).__init__(channel_dim=channel_dim, frame_dim=frame_dim, tab_dim=tab_dim,
                                           hw_size=hw_size, **kwargs)
        if tabular_branch:
            tab_dim = channel_dim
        self.n_tab = tab_dim
        self.tabattention = tabattention
        self.temporal_attention = temporal_attention
        self.cam_sam = cam_sam
        if self.cam_sam:
            self.channel_gate = ChannelGate(channel_dim, reduction_ratio, pool_types, tabattention, tab_dim,
                                            tabular_branch=tabular_branch)
            self.spatial_gate = SpatialGate(tabattention, tab_dim, hw_size, tabular_branch=tabular_branch)
        if temporal_attention:
            self.temporal_gate = TemporalGate(frame_dim, tabattention=tabattention, n_tab=tab_dim,
                                              temporal_mhsa=temporal_mhsa, tabular_branch=tabular_branch)

    def forward(self, x, tab):
        b, c, h, w, f = x.shape
        x_in = torch.permute(x, (0, 4, 1, 2, 3))
        x_in = torch.reshape(x_in, (b * f, c, h, w))
        if self.tabattention:
            tab_rep = tab.repeat(f, 1, 1)
        else:
            tab_rep = None

        if self.cam_sam:
            x_out = self.channel_gate(x_in, tab_rep)
            x_out = self.spatial_gate(x_out, tab_rep)
        else:
            x_out = x_in

        x_out = torch.reshape(x_out, (b, f, c, h, w))

        if self.temporal_attention:
            x_out = self.temporal_gate(x_out, tab)

        x_out = torch.permute(x_out, (0, 2, 3, 4, 1))  # b,c,h,w,f

        return x_out


# _______________ TabAttention ____________________________________________________________________

class MlpBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim=512, output_dim=None):
        super().__init__()
        if output_dim is None:
            output_dim = input_dim
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, output_dim)

    def forward(self, x):
        return self.fc2(self.gelu(self.fc1(x)))


class TabularMLP(TabularModule):
    def __init__(self,
                 bottleneck_tab_dim=16,
                 **kwargs
                 ):
        super(TabularMLP, self).__init__(**kwargs)
        self.bottleneck_dim = bottleneck_tab_dim

        self.tab_embedding_patch = MlpBlock(input_dim=self.tab_dim, mlp_dim=self.bottleneck_dim, output_dim=256)

        self.tab_embedding_channel = MlpBlock(input_dim=self.tab_dim, mlp_dim=self.bottleneck_dim, output_dim=98)

        self.tab_mlp_channel = MlpBlock(input_dim=257, mlp_dim=2048, output_dim=256)
        self.tab_mlp_patch = MlpBlock(input_dim=99, mlp_dim=256, output_dim=98)

        self.norm1 = nn.LayerNorm(256)
        self.norm2 = nn.LayerNorm(256)

    def forward(self, tokens, tab):
        tab = torch.unsqueeze(tab, 1)
        tab_patch = self.tab_embedding_patch(tab)
        tab_channel = self.tab_embedding_channel(tab)
        tab_channel = torch.transpose(tab_channel, 1, 2)

        tokens_norm = self.norm1(tokens)
        x_tab_p = torch.concat([tokens_norm, tab_patch], dim=1)
        x_tab_p = torch.transpose(x_tab_p, 1, 2)
        x_tab_p = self.tab_mlp_patch(x_tab_p)
        x_tab_p = torch.transpose(x_tab_p, 1, 2)

        tokens = x_tab_p + tokens
        tokens_norm = self.norm2(tokens)
        x_tab_c = torch.concat([tokens_norm, tab_channel], dim=2)
        x_tab_c = self.tab_mlp_channel(x_tab_c)
        tokens = x_tab_c + tokens
        return tokens


class SkipConnection(nn.Module):
    """
    Combine the forward pass input with the result from the given submodule::

        --+--submodule--o--
          |_____________|

    The available modes are ``"cat"``, ``"add"``, ``"mul"``.
    """

    def __init__(self, submodule, dim: int = 1, mode: str = "cat") -> None:
        """

        Args:
            submodule: the module defines the trainable branch.
            dim: the dimension over which the tensors are concatenated.
                Used when mode is ``"cat"``.
            mode: ``"cat"``, ``"add"``, ``"mul"``. defaults to ``"cat"``.
        """
        super().__init__()
        self.submodule = submodule
        self.dim = dim
        self.mode = look_up_option(mode, SkipMode).value

    def forward(self, x: torch.Tensor, tab=None) -> torch.Tensor:
        y = self.submodule(x, tab)

        if self.mode == "cat":
            return torch.cat([x, y], dim=self.dim)
        if self.mode == "add":
            return torch.add(x, y)
        if self.mode == "mul":
            return torch.mul(x, y)
        raise NotImplementedError(f"Unsupported mode {self.mode}.")


class DoubleInputSequential(nn.Module):
    def __init__(self, *layers):
        super(DoubleInputSequential, self).__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x, y):
        for l in self.layers:
            if isinstance(l, (SkipConnection, TabularModule, DoubleInputSequential)):
                x = l(x, y)
            else:
                x = l(x)
        return x


class INSIDE(TabularModule):
    def __init__(self,
                 is_return_mask: bool = False,
                 **kwargs
                 ):
        super(INSIDE, self).__init__(**kwargs)
        self.global_pool = nn.AdaptiveAvgPool3d(1)

        layers = [
            ("fc_1", nn.Linear(self.tab_dim, self.channel_dim // 2)),
            ("fc_1_act", nn.Tanh()),
            ("fc_2", nn.Linear(self.channel_dim // 2, self.channel_dim // 2)),
            ("fc_2_act", nn.Tanh()),
        ]
        self.fc_tab = nn.Sequential(OrderedDict(layers))
        self.fc_p = nn.Linear(self.channel_dim // 2, 5 * self.channel_dim)
        self.fc_s = nn.Linear(self.channel_dim // 2, 3 * self.channel_dim)
        self.tanh_mu_a = nn.Tanh()
        self.tanh_mu_b = nn.Tanh()
        self.tanh_mu_c = nn.Tanh()
        self.sig_std = nn.Sigmoid()
        self.is_return_mask = is_return_mask

    def attention(self, mu, sigma, shape, index):
        """1-D Gaussian Attention; one attention per channel.

        :param mu:
            (batch_size, channels) of Gaussian means.

        :param sigma:
            (batch_size, channels) of Gaussian standard deviations (std).

        :param shape:
            Shape (4, ) of input feature maps (batch_size, channels, X, Y, Z).

        :param index:
            Index (int) of the axis to create an attention for.
        """
        if index == 2:
            _arrange = (1, 1, shape[index], 1, 1)
        elif index == 3:
            _arrange = (1, 1, 1, shape[index], 1)
        else:
            _arrange = (1, 1, 1, 1, shape[index])

        x = torch.arange(0, shape[index], device=mu.device).view(_arrange).repeat(shape[0], shape[1], 1, 1, 1).float()

        # Calculate relative coordinates. -> mu <-1;1> -> <-0.5; 5.5> for shape[index]=6
        mu = mu * shape[index] / 2 + shape[index] / 2 - 0.5
        sigma = sigma * 3.5 + torch.finfo(sigma.dtype).eps  # We set the max. width here.

        mu = mu.view(mu.shape[0], mu.shape[1], 1, 1, 1)
        sigma = sigma.view(sigma.shape[0], sigma.shape[1], 1, 1, 1)
        # 1-D Attention.
        mask = torch.exp(-0.5 * ((x - mu) / sigma) ** 2)

        return mask

    def forward(self, x, tab, reg=None):
        scale, shift, mu_a, sigma_a, mu_b, sigma_b, mu_c, sigma_c = self.hypernetwork(tab)
        # print(f"\tC. mod: at scale:{x.shape[2:]}, mean:{scale.mean().item():.4f}, std:{scale.std().item():.4f} max:{scale.max().item():.4f}, min: {scale.min().item():.4f}, shift mean: {shift.mean().item():.4f}.")
        # print(f"\tC. mod: at scale:{x.shape[2:]}, sigma_a mean:{sigma_a.mean().item():.4f}, std:{sigma_a.std().item():.4f}, sigma_b mean:{sigma_b.mean().item():.4f}, std:{sigma_b.std().item():.4f}, sigma_c mean:{sigma_c.mean().item():.4f}, std:{sigma_c.std().item():.4f}")
        scale_values = (scale, shift)
        attention_values = (mu_a, sigma_a, mu_b, sigma_b, mu_c, sigma_c)
        l2 = torch.norm(scale, p=2) + torch.norm(shift, p=2) + torch.norm(sigma_a, p=2) + torch.norm(sigma_b, p=2) + torch.norm(sigma_c, p=2)
        
        # BxCxHxWxD - x,y,z - 1D: eg. a_y = 2x8x1x6x1 
        a_x = self.attention(mu_a, sigma_a, x.shape, 2)
        a_y = self.attention(mu_b, sigma_b, x.shape, 3)
        a_z = self.attention(mu_c, sigma_c, x.shape, 4)

        a_y = a_y.permute(0, 1, 2, 4, 3)
        a_xy = torch.matmul(a_x, a_y).permute(0, 1, 2, 4, 3)
        a = torch.matmul(a_xy, a_z)
        scale = scale.reshape(*scale.shape, 1, 1, 1)
        shift = shift.reshape(*shift.shape, 1, 1, 1)
        
        return (1 - scale) * x * a + shift, l2, (scale_values, attention_values)
            
        
    def hypernetwork(self, tab):
        tab_emb = self.fc_tab(tab)

        p = self.fc_p(tab_emb)  # scale, shift, mean
        s = self.sig_std(self.fc_s(tab_emb))  # std

        scale = p[..., :self.channel_dim]
        shift = p[..., self.channel_dim:self.channel_dim * 2]

        mu_a = self.tanh_mu_a(p[..., self.channel_dim * 2:self.channel_dim * 3])
        sigma_a = s[..., :self.channel_dim]

        mu_b = self.tanh_mu_b(p[..., self.channel_dim * 3:self.channel_dim * 4])
        sigma_b = s[..., self.channel_dim:self.channel_dim * 2]

        mu_c = self.tanh_mu_c(p[..., self.channel_dim * 4:])
        sigma_c = s[..., self.channel_dim * 2:]
        return scale, shift, mu_a, sigma_a, mu_b, sigma_b, mu_c, sigma_c


if __name__ == "__main__":
    import torch

    x = torch.randn(2, 8, 16, 16, 10)
    tab = torch.randn(2, 32)
    model = INSIDE(channel_dim=8, frame_dim=10, hw_size=(16, 16), tab_dim=32)
    y, regl2 = model(x, tab)
    print(y.shape)
