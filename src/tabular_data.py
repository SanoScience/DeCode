import os
import numpy as np
import pandas as pd
from typing import List, Optional, Union
from models.modules import DAFT, FiLM, TabAttention, INSIDE


class RadiomicsDataLoader:
    def __init__(self, path_data, conditional_embeddings = "entity", features="all", feature_selection=False, file_name='radiomics_features_norm.npy'):
        self.radio_shape_features = ['VoxelVolume',
                                     'SurfaceArea',
                                     'SurfaceVolumeRatio',
                                     'Sphericity',
                                     'Compactness1',
                                     'Compactness2',
                                     'SphericalDisproportion',
                                     'Maximum3DDiameter',
                                     'Maximum2DDiameterSlice',
                                     'Maximum2DDiameterColumn',
                                     'Maximum2DDiameterRow',
                                     'MajorAxisLength',
                                     'MinorAxisLength',
                                     'LeastAxisLength',
                                     'Elongation',
                                     'Flatness']
        self.features = features
        self.conditional_embeddings = conditional_embeddings
        self.feature_names_dict = dict(zip(range(len(self.radio_shape_features)), self.radio_shape_features))
        self.features_array = np.load(os.path.join(path_data, file_name))
    
        #set last element of axis as zeros
        self.features_array = np.concatenate((self.features_array, np.zeros((1,)+self.features_array.shape[1:])))
        
        if feature_selection:
            if features == 'all':
                selected_features = self.radio_shape_features
            else:
                self.radio_shape_features = [f for f in self.radio_shape_features if f in selected_features]
                #TODO - allow to choose features and modify the features_array
        
        #setup shape for embeddings
        #patient_visits x tooth_num x features_num -> 97(8)x32x17       
        self.features_dim = self.features_array.shape[-1]
        if len(self.features_array.shape) == 3:
            self.entity_dim = self.features_array.shape[1] * self.features_array.shape[2]
        else:
            self.entity_dim = self.features_array.shape[-1]
        self.create_embeddings()
    
    def __len__(self):
        return self.features_array.shape[0]
    
    def create_embeddings(self):
        if self.conditional_embeddings == "entity":
            self.features_array = self.features_array.reshape(self.__len__(), -1)
            assert self.features_array.shape[-1] == self.entity_dim, "embedding size mismatch"
        elif self.conditional_embeddings == "feature":
            pass
        else:
            raise NotImplementedError(f"There are no implementation of: {args.conditional_embeddings}") 
            

    def get_items(self, ids: Optional[Union[List, np.array]] = None):
        if isinstance(ids, List):
            ids = np.array(ids)
        if any(ids > (self.features_array.shape[0]-1)):
            return np.take(self.features_array, ids, axis=0, mode='clip').astype(np.float32)
        return np.take(self.features_array, ids, axis=0).astype(np.float32)


def create_tabular_config_dict(module, tab_dim, channel_dim, frame_dim, h_dim, w_dim, additional_args=None):
    if additional_args is None:
        return {"module": module, "tab_dim": tab_dim, "channel_dim": channel_dim, "frame_dim": frame_dim,
                "hw_size": (h_dim, w_dim)}
    else:
        conf = {"module": module, "tab_dim": tab_dim, "channel_dim": channel_dim, "frame_dim": frame_dim,
                "hw_size": (h_dim, w_dim)}
        return {**conf, **additional_args}

def create_tabular_config_dict(module, tab_dim, channel_dim, frame_dim, h_dim, w_dim, additional_args=None):
    if additional_args is None:
        return {"module": module, "tab_dim": tab_dim, "channel_dim": channel_dim, "frame_dim": frame_dim,
                "hw_size": (h_dim, w_dim)}
    else:
        conf = {"module": module, "tab_dim": tab_dim, "channel_dim": channel_dim, "frame_dim": frame_dim,
                "hw_size": (h_dim, w_dim)}
        return {**conf, **additional_args}

def model_size_config(model_config_name: str = 'M1'):
    nb = 1
    fc = {'S': 8, 'M': 16, 'L': 32, 'XL': 64, 'XXL': 64}[model_config_name[:-1]]
    mlp_ratio = float(model_config_name[-1])
    if model_config_name[:-1] == 'XXL':
        nb = 2
    embed_dims = [8 * fc, 10 * fc, 16 * fc]
    return nb, fc, mlp_ratio, embed_dims

def get_tabular_config(tab_dim, args, additional_args=None):
    h, w, d = args.patch_size[0], args.patch_size[1], args.patch_size[2]

    if args.model_name == "UNet":
        feature_maps = tuple(2 ** i * args.n_features for i in range(0, args.unet_depth))
        scale_factor = 2**(args.unet_depth-1)
        return create_tabular_config_dict(module=args.tabular_module, tab_dim=tab_dim, channel_dim=feature_maps[-1],
                                          frame_dim=d // scale_factor, h_dim=h // scale_factor, w_dim=w // scale_factor, additional_args=additional_args)
    elif  "NeXt3D" in args.model_name:
        nb, fc, mlp_ratio, embed_dims = model_size_config(args.model_config)
        scale_factor = 2**5
        return create_tabular_config_dict(module=args.tabular_module, tab_dim=tab_dim, channel_dim=16 * fc,
                                          frame_dim=d // scale_factor,
                                          h_dim=h // scale_factor,
                                          w_dim=w // scale_factor, additional_args=additional_args)
    elif args.model_name == 'AE':
        scale_factor = 2**args.unet_depth
        fc = args.n_features
        return create_tabular_config_dict(module=args.tabular_module, tab_dim=tab_dim, channel_dim=16 * fc,
                                          frame_dim=d // scale_factor,
                                          h_dim=h // scale_factor,
                                          w_dim=w // scale_factor, additional_args=additional_args)
    else:
        return None
        # raise NotImplementedError(f"No Implementation of tabular modules for model: {args.model_name}")

def get_module_from_config(conf):
    if conf["module"] == "DAFT":
        return DAFT(**conf)
    elif conf["module"] == "FiLM":
        return FiLM(**conf)
    elif conf["module"] == "INSIDE":
        return INSIDE(**conf)
    else:
        raise NotImplementedError(
            f"There is no implementation of: {conf.module}")

if __name__ == '__main__':
    rdl = RadiomicsDataLoader('data/radiomics/china')
    print(rdl.get_items([1,24,23]))
    
