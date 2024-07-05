import torch
import torch.nn.functional as F

__all__ = ['UNext3D']

from timm.models.layers import DropPath, to_3tuple, trunc_normal_
import math

import argparse
import torch.nn as nn


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    from modules import TabAttention, TabularMLP
    from tabular_data import get_module_from_config, get_tabular_config
else:
    from .modules import TabAttention, TabularMLP
    from tabular_data import get_module_from_config


class qkv_transform(nn.Conv1d):
    """Conv1d for qkv_transform"""


def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, bias=False)


class shiftmlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., shift_size=5):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.shift_size = shift_size
        self.pad = shift_size // 2

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, D):
        B, N, C = x.shape

        xn = x.transpose(1, 2).view(B, C, H, W, D).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 2) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = torch.narrow(x_s, 4, self.pad, D)

        x_s = x_s.reshape(B, C, H * W * D).contiguous()
        x_shift_r = x_s.transpose(1, 2)

        x = self.fc1(x_shift_r)

        B, N, C = x.shape  # my mod

        x = self.dwconv(x, H, W, D)
        x = self.act(x)
        x = self.drop(x)

        xn = x.transpose(1, 2).view(B, C, H, W, D).contiguous()
        xn = F.pad(xn, (self.pad, self.pad, self.pad, self.pad, self.pad, self.pad), "constant", 0)
        xs = torch.chunk(xn, self.shift_size, 1)
        x_shift = [torch.roll(x_c, shift, 3) for x_c, shift in zip(xs, range(-self.pad, self.pad + 1))]
        x_cat = torch.cat(x_shift, 1)
        x_cat = torch.narrow(x_cat, 2, self.pad, H)
        x_s = torch.narrow(x_cat, 3, self.pad, W)
        x_s = torch.narrow(x_s, 4, self.pad, D)
        x_s = x_s.reshape(B, C, H * W * D).contiguous()
        x_shift_c = x_s.transpose(1, 2)

        x = self.fc2(x_shift_c)
        x = self.drop(x)
        return x


class shiftedBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4., drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = shiftmlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, D):

        x = x + self.drop_path(self.mlp(self.norm2(x), H, W, D))
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv3d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W, D):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W, D)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_3tuple(patch_size)

        self.patch_size = patch_size
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2, patch_size[2] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
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
        x = self.proj(x)
        _, _, H, W, D = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W, D


def model_size_config(model_config_name: str = 'M1'):
    nb = 1
    fc = {'S': 8, 'M': 16, 'L': 32, 'XL': 64, 'XXL': 64}[model_config_name[:-1]]
    mlp_ratio = float(model_config_name[-1])
    if model_config_name[:-1] == 'XXL':
        nb = 2
    embed_dims = [8 * fc, 10 * fc, 16 * fc]
    return nb, fc, mlp_ratio, embed_dims


class UNext3D(nn.Module):

    ## Conv 3 + MLP 2 + shifted MLP

    def __init__(self, num_classes, img_size=(224, 224, 224), in_chans=1,
                 mlp_ratio: float = 1.0, drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[1, 1, 1], model_config: str = 'M1', tabular_config=None, tabular_module=None, **kwargs):
        super().__init__()

        # MODEL SIZE CONFIG
        nb, fc, mlp_ratio, embed_dims = model_size_config(model_config_name=model_config)

        self.encoder1 = nn.Conv3d(in_chans, fc, 3, stride=1, padding=1)  # in_chans, 16
        self.encoder2 = nn.Conv3d(fc, 2 * fc, 3, stride=1, padding=1)  # 16, 32
        self.encoder3 = nn.Conv3d(2 * fc, 8 * fc, 3, stride=1, padding=1)  # 32, 128

        self.ebn1 = nn.BatchNorm3d(fc)  # 16
        self.ebn2 = nn.BatchNorm3d(2 * fc)  # 32
        self.ebn3 = nn.BatchNorm3d(8 * fc)  # 128

        self.norm3 = norm_layer(embed_dims[1])  # 160
        self.norm4 = norm_layer(embed_dims[2])  # 256

        self.dnorm3 = norm_layer(embed_dims[1])  # 160
        self.dnorm4 = norm_layer(embed_dims[0])  # 128

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.block1 = nn.ModuleList(
            nb * [shiftedBlock(dim=embed_dims[1], mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[0],
                               norm_layer=norm_layer)])

        self.block2 = nn.ModuleList(
            nb * [shiftedBlock(dim=embed_dims[2], mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[1],
                               norm_layer=norm_layer)])

        self.dblock1 = nn.ModuleList(
            nb * [shiftedBlock(dim=embed_dims[1], mlp_ratio=mlp_ratio, drop=drop_rate, drop_path=dpr[0],
                               norm_layer=norm_layer)])

        self.dblock2 = nn.ModuleList(
            nb * [
                shiftedBlock(dim=embed_dims[0], mlp_ratio=1, drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer)])

        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])

        self.decoder1 = nn.Conv3d(16 * fc, 10 * fc, 3, stride=1, padding=1)  # 256, 160
        self.decoder2 = nn.Conv3d(10 * fc, 8 * fc, 3, stride=1, padding=1)  # 160, 128
        self.decoder3 = nn.Conv3d(8 * fc, 2 * fc, 3, stride=1, padding=1)  # 128, 32
        self.decoder4 = nn.Conv3d(2 * fc, fc, 3, stride=1, padding=1)  # 32, 16
        self.decoder5 = nn.Conv3d(fc, fc, 3, stride=1, padding=1)  # 16, 16

        self.dbn1 = nn.BatchNorm3d(10 * fc)  # 160
        self.dbn2 = nn.BatchNorm3d(8 * fc)  # 128
        self.dbn3 = nn.BatchNorm3d(2 * fc)  # 32
        self.dbn4 = nn.BatchNorm3d(fc)  # 16
        
        self.upsample = nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear', align_corners=False)

        self.final = nn.Conv3d(fc, num_classes, kernel_size=1)
        self.soft = nn.Softmax(dim=1)
        if tabular_config is not None:
            self.tabular_config = tabular_config
            self.tabular_module = get_module_from_config(self.tabular_config)
        else:
            if tabular_module is not None:
                self.tabular_module = tabular_module
            else:
                self.tabular_module = None

    def forward(self, x, tab=None):
        B, C_orig, H_orig, W_orig, D_orig = x.shape

        ### Encoder
        ### Conv Stage

        ### Stage 1
        out = torch.relu(F.max_pool3d(self.ebn1(self.encoder1(x)), 2, 2))
        t1 = out
        ### Stage 2
        out = torch.relu(F.max_pool3d(self.ebn2(self.encoder2(out)), 2, 2))
        t2 = out
        ### Stage 3
        out = torch.relu(F.max_pool3d(self.ebn3(self.encoder3(out)), 2, 2))
        t3 = out

        ### Tokenized MLP Stage
        ### Stage 4

        out, H, W, D = self.patch_embed3(out)
        for i, blk in enumerate(self.block1):
            out = blk(out, H, W, D)
        out = self.norm3(out)
        out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
        t4 = out

        ### Bottleneck (C5)

        out, H, W, D = self.patch_embed4(out)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W, D)
        out = self.norm4(out)

        if tab is not None:
            if self.tabular_module:
                if isinstance(self.tabular_module, TabularMLP):
                    out = self.tabular_module(out, tab)
                    out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
                else:
                    out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
                    out = self.tabular_module(out, tab)
        else:
            out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()

        ### Stage 4
        out = torch.relu(self.upsample(self.dbn1(self.decoder1(out))))

        out = torch.add(out, t4)
        _, _, H, W, D = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W, D)

        ### Stage 3

        out = self.dnorm3(out)
        out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()
        out = torch.relu(self.upsample(self.dbn2(self.decoder2(out))))
        out = torch.add(out, t3)
        _, _, H, W, D = out.shape
        out = out.flatten(2).transpose(1, 2)

        for i, blk in enumerate(self.dblock2):
            out = blk(out, H, W, D)

        out = self.dnorm4(out)
        out = out.reshape(B, H, W, D, -1).permute(0, 4, 1, 2, 3).contiguous()

        out = torch.relu(self.upsample(self.dbn3(self.decoder3(out))))
        out = torch.add(out, t2)
        out = torch.relu(self.upsample(self.dbn4(self.decoder4(out))))
        out = torch.add(out, t1)
        out = torch.relu(self.upsample((self.decoder5(out))))

        out = self.final(out)
        return out


if __name__ == "__main__":
    # conf = {"channel_dim": 128, "frame_dim": 6, "hw_size": (224 // 8, 224 // 8), "tab_dim": 6}
    # model = UNext3D(num_classes=1, in_chans=1, model_config='XXL4', tabular_module=None)
    # # model = UNext3D(num_classes=1, tabular_module=None)
    # input = torch.rand(1, 1, 224, 224, 160)
    # tab = torch.rand(1, 6)
    # output = model(input, tab)
    # # output = model(input)
    # print(output.shape)
    
    import os
    import random
    import time
    import numpy as np
    from monai.utils import set_determinism
    
    # seed=48
    device = torch.device("cuda", index=0) 
    # torch.cuda.set_device(device)
    # NP_MAX = np.iinfo(np.uint32).max
    # MAX_SEED = NP_MAX + 1 
    # os.environ["PYTHONHASHSEED"] = str(seed)
    # seed = int(seed) % MAX_SEED
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # random.seed(seed)
    # np.random.seed(seed)
    
    # #SLOW deterministic chains
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    # os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
    # torch.use_deterministic_algorithms(mode=True, warn_only=False)
        
    # conf = {"channel_dim": 128, "frame_dim": 6, "hw_size": (224 // 8, 224 // 8), "tab_dim": 6}
    # model = UNext3D(num_classes=1, in_chans=1, model_config='M1', tabular_module=None).to(device)
    # img = torch.rand(1, 1, 224, 224, 160).to(device)
    # tab = torch.rand(1, 6)
    # time_list = []
    # for i in range(300):
    #     start = time.time()
    #     output = model(img, tab)
    #     torch.cuda.synchronize(device)
    #     end = time.time()-start
    #     if i > 10:
    #         time_list.append(end)
      
    # print(output.shape)
    # print(f"avg. forward time: {np.array(time_list).mean()*1000:.3f}ms.")
    
    import yaml
    import os
    from argparse import Namespace

    config_dir = os.path.join('config', 'general_config.yaml')
    with open(config_dir, 'r') as file:
        config = yaml.safe_load(file)
    args = Namespace(**config['args'])
    # conf = get_tabular_config(tab_dim=6, args=args)
    model = UNext3D(num_classes=1, in_chans=1, model_config=args.model_config)
    # print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable paramters: {pytorch_trainable_params}, all parameters: {pytorch_total_params}.")
    # model = UNext3D(num_classes=1, tabular_module=None)
    input = torch.rand(2, 1, 224, 224, 160)
    tab = torch.rand(2, 6)
    output = model(input, None)
    # output = model(input)
    print(output.shape)
