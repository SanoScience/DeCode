import torch
import torch.nn.functional as F
import argparse
import numpy as np
import torch.nn as nn
from typing import Optional, Sequence, Tuple, Union
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Act, Norm
from itertools import chain
import warnings

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.join(os.getcwd(), 'src'))
    from tabular_data import get_module_from_config
    from modules import DAFT, FiLM, TabAttention, TabularModule, INSIDE
else:
    from .modules import DAFT, FiLM, TabAttention, TabularModule, INSIDE
    # from tabular_data import get_module_from_config
    
def get_conditioning_dict(tabular_module_name):
    if tabular_module_name in ['FiLM', 'DAFT']: 
        keys = [[f'{param}_{val}' for val in ['min', 'max', 'mean', 'std']] for param in ['a', 'b']]
        keys = ['decoder_stage'] +  list(chain(*keys))
        conditioning_dict = {k:[] for k in keys}
        return conditioning_dict
    if tabular_module_name == 'INSIDE':
        keys_scale = [[f'{param}_{val}' for val in ['min', 'max', 'mean', 'std']] for param in ['a', 'b']]
        keys_attention_sigma =[[f'sigma_{param}_{val}' for val in ['min', 'max', 'mean', 'std']] for param in ['x', 'y', 'z']]
        keys_attention_mean = [[f'mean_{param}_{val}' for val in ['min', 'max', 'mean', 'std']] for param in ['x', 'y', 'z']]
        keys = ['decoder_stage'] +  list(chain(*keys_scale)) + list(chain(*keys_attention_sigma)) + list(chain(*keys_attention_mean))
        conditioning_dict = {k:[] for k in keys}
        return conditioning_dict
    
def calculate_conditioning_stats(conditioning_dict, decoder_stage, scale, attention = None):
    conditioning_dict['decoder_stage'].append(decoder_stage)
    for idx, val in enumerate(['a', 'b']):
        conditioning_dict[f'{val}_min'].append(scale[idx].min().item())
        conditioning_dict[f'{val}_max'].append(scale[idx].max().item())
        conditioning_dict[f'{val}_mean'].append(scale[idx].mean().item())
        conditioning_dict[f'{val}_std'].append(scale[idx].std().item())
    if attention is not None:
        for idx, val in enumerate(['mean_x', 'sigma_x', 'mean_y', 'sigma_y', 'mean_z', 'sigma_z']):
            conditioning_dict[f'{val}_min'].append(attention[idx].min().item())
            conditioning_dict[f'{val}_max'].append(attention[idx].max().item())
            conditioning_dict[f'{val}_mean'].append(attention[idx].mean().item())
            conditioning_dict[f'{val}_std'].append(attention[idx].std().item())
            

class DeCode(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[Sequence[int], int] = 3,
        up_kernel_size: Union[Sequence[int], int] = 2,
        act: Union[Tuple, str] = Act.RELU,
        norm: Union[Tuple, str] = Norm.BATCH,
        bias: bool = False,
        adn_ordering: str = "NDA",
        tabular_module: Optional[str] = 'FiLM',
        embedding_size: int = 512,
        module_bottleneck_dim: int = 128,
        is_regression: bool = True,
        is_unet_skip: bool = True,
        is_inference_regression: bool = False,
        is_log_conditioning: bool = True,
        regression_mlp_expansion: int = 1,
        is_embedding = True,
        is_inference_embedding = True
    ) -> None:
        super().__init__()
       
        self.is_embedding = is_embedding
        self.bottleneck_dim = module_bottleneck_dim
        if tabular_module is None:
            self.is_condition  = False
        else:
            self.is_condition  = True
            self.tabular_module_name = tabular_module
            if tabular_module == 'FiLM':
                tabular_module = FiLM 
            elif tabular_module == 'INSIDE':
                tabular_module = INSIDE 
            elif tabular_module == 'DAFT':
                tabular_module = DAFT
            
            
        self.is_regression = is_regression
        self.is_inference_regression = is_inference_regression
        self.is_inference_embedding = is_inference_embedding
        self.is_unet_skip = is_unet_skip
        self.is_log_conditioning = is_log_conditioning
        
        #Encoder
        self.e1 = Convolution(spatial_dims=3, in_channels=in_channels, out_channels=16, strides=1, kernel_size=kernel_size,
                                        adn_ordering=adn_ordering, norm=norm, act=act, bias=bias)
        self.e1_2 = nn.Conv3d(in_channels=16, out_channels=16, stride=2, kernel_size=kernel_size, padding=1, bias=bias)
        self.e1_2NA = nn.Sequential(nn.BatchNorm3d(16), nn.ReLU())
        
        self.e2 = Convolution(spatial_dims=3, in_channels=16, out_channels=32, strides=1, kernel_size=kernel_size,
                                adn_ordering=adn_ordering, norm=norm, act=act, bias=bias)
        self.e2_2 = nn.Conv3d(in_channels=32, out_channels=32, stride=2, kernel_size=kernel_size, padding=1, bias=bias)
        self.e2_2NA = nn.Sequential(nn.BatchNorm3d(32), nn.ReLU())
        
        self.e3 = Convolution(spatial_dims=3, in_channels=32, out_channels=64, strides=1, kernel_size=kernel_size,
                                adn_ordering=adn_ordering, norm=norm, act=act, bias=bias)
        self.e3_2 = nn.Conv3d(in_channels=64, out_channels=64, stride=2, kernel_size=kernel_size, padding=1, bias=bias)
        self.e3_2NA = nn.Sequential(nn.BatchNorm3d(64), nn.ReLU())
        
        self.e4 = Convolution(spatial_dims=3, in_channels=64, out_channels=128, strides=1, kernel_size=kernel_size,
                                adn_ordering=adn_ordering, norm=norm, act=act, bias=bias)
        self.e4_2 = nn.Conv3d(in_channels=128, out_channels=128, stride=2, kernel_size=kernel_size, padding=1, bias=bias)
        self.e4_2NA = nn.Sequential(nn.BatchNorm3d(128), nn.ReLU())
        
        self.e5 = Convolution(spatial_dims=3, in_channels=128, out_channels=256, strides=1, kernel_size=kernel_size,
                              adn_ordering=adn_ordering, norm=norm, act=act, bias=bias)
        
        #upconv padding
        pad = up_kernel_size - 2
        
        self.d5_NA = nn.Sequential(nn.BatchNorm3d(128), nn.ReLU())
        self.d4_NA = nn.Sequential(nn.BatchNorm3d(64), nn.ReLU())
        self.d3_NA = nn.Sequential(nn.BatchNorm3d(32), nn.ReLU())
        self.d2_NA = nn.Sequential(nn.BatchNorm3d(16), nn.ReLU())
        
        #Decoder layers, di_2 correponds to 2x upsampling with transposed conv
        self.d5 = Convolution(spatial_dims=3, in_channels=256, out_channels=128, strides=1, kernel_size=kernel_size,
                                adn_ordering=adn_ordering, norm=None, act=None, bias=bias)
        if self.is_condition:
            self.conditioning_layer_d5 = tabular_module(channel_dim=128, tab_dim=embedding_size, bottleneck_dim=self.bottleneck_dim)
        self.d5_2 = Convolution(spatial_dims=3, in_channels=128, out_channels=128, strides=2, kernel_size=up_kernel_size, act=act, norm=Norm.BATCH, is_transposed=True, padding=pad, output_padding=pad, dilation=1, bias=bias)
        
        self.d4 = Convolution(spatial_dims=3, in_channels=128, out_channels=64, strides=1, kernel_size=kernel_size,
                                adn_ordering=adn_ordering, norm=None, act=None, bias=bias)
        self.d4_2 = Convolution(spatial_dims=3, in_channels=64, out_channels=64, strides=2, kernel_size=up_kernel_size, act=act, norm=Norm.BATCH, is_transposed=True, padding=pad, output_padding=pad, dilation=1, bias=bias)
        if self.is_condition:
            self.conditioning_layer_d4 = tabular_module(channel_dim=64, tab_dim=embedding_size, bottleneck_dim=self.bottleneck_dim)
        
        self.d3 = Convolution(spatial_dims=3, in_channels=64, out_channels=32, strides=1, kernel_size=kernel_size,
                        adn_ordering=adn_ordering, norm=None, act=None, bias=bias)
        self.d3_2 = Convolution(spatial_dims=3, in_channels=32, out_channels=32, strides=2, kernel_size=up_kernel_size, act=act, norm=Norm.BATCH, is_transposed=True, padding=pad, output_padding=pad, dilation=1, bias=bias)
        if self.is_condition:
            self.conditioning_layer_d3 = tabular_module(channel_dim=32, tab_dim=embedding_size, bottleneck_dim=self.bottleneck_dim)
        
        self.d2 = Convolution(spatial_dims=3, in_channels=32, out_channels=16, strides=1, kernel_size=kernel_size,
                adn_ordering=adn_ordering, norm=None, act=None, bias=bias)
        self.d2_2 = Convolution(spatial_dims=3, in_channels=16, out_channels=16, strides=2, kernel_size=up_kernel_size, act=act, norm=Norm.BATCH, is_transposed=True, padding=pad, output_padding=pad, dilation=1, bias=bias)
        if self.is_condition:
            self.conditioning_layer_d2 = tabular_module(channel_dim=16, tab_dim=embedding_size, bottleneck_dim=self.bottleneck_dim)
        
        self.d1 = nn.Conv3d(in_channels=16, out_channels=out_channels, kernel_size=1, bias=bias)
        
        #regression and embedding MLPs
        if self.is_regression:
            self.gap = nn.AvgPool3d(kernel_size=(14,14,10))
            if self.is_embedding:
                # encoder (e5) image features embedding-based conditioning
                # WEIGHTED EMBEDDING
                self.image_conditioning_embedding = nn.Sequential(nn.Linear(in_features=256, out_features=embedding_size, bias=False))
        
                # input is e5-based embedding (image_conditioning_net output)
                self.shape_features_regression_net = nn.Sequential(nn.ReLU(),
                                                                   nn.Linear(in_features=embedding_size, out_features=embedding_size, bias=False))
                # shape features embedding-based conditioning
                self.shape_features_conditioning_net = nn.Sequential(nn.Linear(in_features=embedding_size, out_features=int(embedding_size*regression_mlp_expansion), bias=False),
                                                                     nn.ReLU(),
                                                                     nn.Linear(in_features=int(embedding_size*regression_mlp_expansion), out_features=embedding_size, bias=False)
                                                                    )
            else:
                #shape features regression result-based conditioning
                #input directly from e5
                self.shape_features_regression_net = nn.Sequential(nn.Linear(in_features=256, out_features=int(embedding_size*regression_mlp_expansion), bias=False),
                                                                   nn.ReLU(),
                                                                   nn.Linear(in_features=int(embedding_size*regression_mlp_expansion), out_features=embedding_size, bias=False))
              
    def forward(self, x=None, tab=None):
        y_feature_regression = None
        imaging_embedding = None
        conditioning_embedding = None
        cond_stats = None
        
        if self.is_condition:
            reg_termL2 = torch.zeros(1, device = x.device)
            if self.is_log_conditioning:
                cond_stats = get_conditioning_dict(self.tabular_module_name)
        else:
            reg_termL2 = None

        #encoder
        # detach feature extraction and batch (N)orm + (A)ctivation  
        if self.is_condition:
            e1 = self.e1_2(self.e1(x))
            e2 = self.e2_2(self.e2(self.e1_2NA(e1)))
            e3 = self.e3_2(self.e3(self.e2_2NA(e2)))
            e4 = self.e4_2(self.e4(self.e3_2NA(e3)))
            e5 = self.e5(self.e4_2NA(e4))
        else:
            e1 = self.e1_2NA(self.e1_2(self.e1(x)))
            e2 = self.e2_2NA(self.e2_2(self.e2(e1)))
            e3 = self.e3_2NA(self.e3_2(self.e3(e2)))
            e4 = self.e4_2NA(self.e4_2(self.e4(e3)))
            e5 = self.e5(e4)
        
        #bottleneck - regression, embedding
        if self.is_regression:
            encoder_features_e5 = self.gap(e5).view(x.shape[0],-1)
            #learn imaging feature embedding L1 simmilar to shape features-based embedding to perform conditioning
            if self.is_embedding:
                # imaging_embedding = self.image_conditioning_net(encoder_features_e5)
                imaging_embedding = self.image_conditioning_embedding(encoder_features_e5)
                #TRAIN
                if self.training:
                    y_feature_regression = self.shape_features_regression_net(imaging_embedding)
                    conditioning_embedding = self.shape_features_conditioning_net(tab)
                    tab = conditioning_embedding
                #INFERENCE
                else:
                    #imaging-based learned embedding (LESF)
                    if self.is_inference_embedding:
                        tab = imaging_embedding
                    #calculated shape features (CSF) embedding
                    else:
                        conditioning_embedding = self.shape_features_conditioning_net(tab)
                        tab = conditioning_embedding
            else:
                y_feature_regression = self.shape_features_regression_net(encoder_features_e5)
                if self.is_inference_regression and not self.training:
                    tab = torch.sigmoid(y_feature_regression)
            
        #decoder
        #stage4 - 128 channels
        d5 = self.d5(e5)
        if self.is_condition:
            if self.is_unet_skip:
                d5_skip = torch.add(d5, e4)
            d5_cond, l2, (scale, attention) = self.conditioning_layer_d5(d5_skip, tab)
            if self.is_log_conditioning:
                calculate_conditioning_stats(cond_stats, 'd5', scale, attention)
            reg_termL2+= l2 / 8
            d5 = torch.add(self.d5_NA(d5), torch.relu(d5_cond))
        else:
            d5=self.d5_NA(d5)
            if self.is_unet_skip:
                d5 = torch.add(d5, e4)
        d4 = self.d5_2(d5)
        
        #stage3 - 64 channels
        d4 = self.d4(d4)
        if self.is_condition:
            if self.is_unet_skip:
                d4_skip = torch.add(d4, e3)
            d4_cond, l2, (scale, attention) = self.conditioning_layer_d4(d4_skip, tab)
            if self.is_log_conditioning:
                calculate_conditioning_stats(cond_stats, 'd4', scale, attention)
            reg_termL2+= l2 / 4
            d4 = torch.add(self.d4_NA(d4), torch.relu(d4_cond))
        else:
            d4=self.d4_NA(d4)
            if self.is_unet_skip:
                d4 = torch.add(d4, e3)
        d3 = self.d4_2(d4)
        
        #stage2 - 32 channels
        d3 = self.d3(d3) 
        if self.is_condition:
            if self.is_unet_skip:
                d3_skip = torch.add(d3, e2)
            d3_cond, l2, (scale, attention) = self.conditioning_layer_d3(d3_skip, tab)
            if self.is_log_conditioning:
                calculate_conditioning_stats(cond_stats, 'd3', scale, attention)
            reg_termL2+= l2 / 2
            d3 = torch.add(self.d3_NA(d3), torch.relu(d3_cond))
        else:
            d3=self.d3_NA(d3)
            if self.is_unet_skip:
                d3 = torch.add(d3, e2)
        d2 = self.d3_2(d3)
        
        #stage1 - 16 channels
        d2 = self.d2(d2) 
        if self.is_condition:
            if self.is_unet_skip:
                d2_skip = torch.add(d2, e1)
            d2_cond, l2, (scale, attention) = self.conditioning_layer_d2(d2_skip, tab)
            if self.is_log_conditioning:
                calculate_conditioning_stats(cond_stats, 'd2', scale, attention)
            reg_termL2+= l2
            d2 = torch.add(self.d2_NA(d2), torch.relu(d2_cond))
        else:
            d2=self.d2_NA(d2)
            if self.is_unet_skip:
                d2 = torch.add(d2, e1)
        d1 = self.d2_2(d2) 
        
        #final layer
        out = self.d1(d1)
        if self.training:
            return out, reg_termL2, y_feature_regression, cond_stats, imaging_embedding, conditioning_embedding
        else:
            return out
    
if __name__ == "__main__":
    import time 
    import numpy as np
    
    model = DeCode(spatial_dims=3,
                      in_channels=1,
                      out_channels=1,
                      act='relu',
                      norm='batch',
                      bias=False,
                      up_kernel_size = 2,
                      tabular_module='FiLM',
                      embedding_size=512,
                      module_bottleneck_dim=128,
                      is_regression=True,
                      is_unet_skip=True,
                      is_inference_regression=True
                    )
    
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"trainable paramters: {pytorch_trainable_params}, all parameters: {pytorch_total_params}.")
    
    device="cuda:0"
    model=model.to(device)
    img = torch.rand(1, 1, 224, 224, 160).to(device)
    tab = torch.rand(1, 544).to(device)
    time_list = []
    model.train()
    for i in range(30):
        start = time.time()
        output = model(img, tab)
        torch.cuda.synchronize(device)
        end = time.time()-start
        if i > 10:
            time_list.append(end)
    
    # print(output.shape)
    print(f"avg. forward time: {np.array(time_list).mean()*1000:.3f}ms.")
    
