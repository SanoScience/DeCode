import os
import sys
import torch
import itertools
import numpy as np
from argparse import Namespace
from typing import Optional

from monai.transforms import (
    AsDiscrete,
    Activations,
    Compose,
    ToDevice)
    
from monai.transforms.utils_pytorch_numpy_unification import clip, percentile

from monai.transforms import (
    EnsureChannelFirstD,
    EnsureTypeD,
    LoadImageD,
    LambdaD,
    NormalizeIntensityD,
    OrientationD,
    RandGaussianNoiseD,
    RandGaussianSmoothD,
    RandShiftIntensityD,
    RandAffineD,
    RandRotateD,
    RandFlipD,
    ScaleIntensityD,
    ScaleIntensityRangeD,
    ScaleIntensityRangePercentilesD,
    SpacingD,
    ThresholdIntensityD,
    ToDeviceD,
)
if os.path.basename(os.getcwd()) == 'UNext':
    sys.path.append(os.path.join(os.getcwd(), 'src'))
from crop_foreground import CropForegroundFixedD


class Transforms():
    def __init__(self,
                 args : Optional[Namespace] = None,
                 device: str = 'cpu',
                 **kwargs
                 ) -> None:
        if args is None:
            args = Namespace(**kwargs)
            default_params = {"keys": ["image", "label"], "pixdim": 0.4, "classes": 32,
                              "z_score_norm": False, "percentile_clip": False, "use_train_augmentations": True,
                              "houndsfield_clip": 3500, 'spatial_crop_margin': (32,)*3, 'spatial_crop_size': (256,)*3,
                              "patch_size": (128,)*3, "lazy_resampling": True, "crop_samples": 1, "rotation_range": 0.1,
                              "translate_range": 0.1, "seed": -1, "crop_foreground": True}
            for k, v in default_params.items():
                if not vars(args).get(k, False):
                    vars(args)[k] = v
        
        #multiclass = teeth classes(32) + background
        self.class_treshold = args.classes if args.classes == 1 else args.classes - 1
        #avoid 'radiomics' key
        keys = args.keys[:2]
        pixdim = (args.pixdim,)*3
        interpolation_dict = {"image": "bilinear", "label": "nearest"}
        mode = [interpolation_dict[k] for k in keys]
        
        if args.percentile_clip:
            self.intensity_preprocessing = [
                #clip to range <0;1> -> HU: <0;99.5 percentile clip>
                ScaleIntensityRangePercentilesD(keys="image",
                                                lower=0.5,
                                                upper=99.5,
                                                b_min=0,
                                                b_max=1,
                                                clip=True),
                ThresholdIntensityD(keys=["label"], above=False, threshold=self.class_treshold,
                                    cval=self.class_treshold),
                EnsureTypeD(keys=keys, data_type="tensor", device=device)
            ]
        elif args.z_score_norm:
             self.intensity_preprocessing = [
                LambdaD(keys="image", 
                        func=lambda x: clip(x, a_min=percentile(x, 0.5), a_max=percentile(x, 99.5))),
                #calculate mean and std per image
                NormalizeIntensityD(keys="image",
                                    subtrahend=None,
                                    divisor=None),
                ScaleIntensityD(keys="image", minv=0, maxv=1)
            ]
        else:
             self.intensity_preprocessing = [
                ScaleIntensityRangeD(keys="image",
                                     a_min=0,
                                     a_max=args.houndsfield_clip,
                                     b_min=0.0,
                                     b_max=1.0,
                                     clip=True),
                ThresholdIntensityD(keys=["label"], above=False, threshold=self.class_treshold,
                                    cval=self.class_treshold),
                EnsureTypeD(keys=keys, data_type="tensor", device=device)
            ]
        self.data_loading = [
            LoadImageD(keys=keys, reader='NibabelReader',
                       image_only=False),
            EnsureChannelFirstD(
                keys=keys, channel_dim='no_channel'),
            OrientationD(keys=keys, axcodes="RAS"),
            ToDeviceD(keys=keys, device=device),
            EnsureTypeD(
                keys=keys, data_type="tensor", device=device),
            SpacingD(keys=keys, pixdim=pixdim,
                     mode=mode)]
        if args.crop_foreground:
           self.data_loading.append(CropForegroundFixedD(keys=keys,
                                 source_key="label",
                                 select_fn=lambda x: x > 0,
                                 margin=args.spatial_crop_margin,
                                 spatial_size=args.patch_size,
                                 mode='constant',
                                 return_coords=True))
        self.data_augmentation = [
            RandShiftIntensityD(keys="image", offsets=0.2, prob=1.0),
            RandGaussianNoiseD(keys="image", prob=1.0, mean=0, std=1.0),
            RandGaussianSmoothD(keys="image", sigma_x=(0.1, 0.5), sigma_y=(0.1, 0.5), sigma_y=(0.1, 0.5), prob=1.0),
            # GEOMETRIC - RANDOM - DATA AUGMENTATION
            # lazy resampling operations
            RandFlipD(keys=keys, prob=1.0, spatial_axis=(0, 1, 2), lazy=args.lazy_resampling),
            RandRotateD(keys=keys, range_x=args.rotation_range, range_y=args.rotation_range, range_z=args.rotation_range,
                        mode=mode, prob=1.0, lazy=args.lazy_resampling),
            RandAffineD(keys=keys, translate_range=(args.translate_range,)*3, mode=mode, padding_mode='reflection', lazy=args.lazy_resampling, p=1.0),
            EnsureTypeD(keys=keys, data_type="tensor", device=device)
        ]
        
        if not args.use_train_augmentations:
            self.data_augmentation = []

        self.train_transform = Compose(self.data_loading + self.intensity_preprocessing + self.data_augmentation)
        self.val_transform = Compose(self.data_loading + self.intensity_preprocessing)
        
        if args.seed != -1:
            state = np.random.RandomState(seed=args.seed)
            self.train_transform.set_random_state(seed=args.seed, state=state)
        
        if args.classes > 1:
            self.post_pred_train = Compose([Activations(softmax=True, dim=1),
                                            AsDiscrete(argmax=True,
                                                       dim=1,
                                                       keepdim=True)
                                            ])
            self.post_pred = Compose([Activations(softmax=True, dim=0),
                                      AsDiscrete(argmax=True,
                                                 dim=0,
                                                 keepdim=True),
                                      ToDevice(device=device)
                                      ])
            self.post_pred_labels = Compose([AsDiscrete(argmax=False,
                                                        to_onehot=args.classes,
                                                        dim=0),
                                            ToDevice(device=device)
                                             ])
        elif args.classes == 1:
            self.post_pred = Compose([Activations(sigmoid=True),
                                      AsDiscrete(threshold=0.5)],
                                     ToDevice(device=device))

        self.none_transform = Compose(
            [
                # INITAL SETUP
                LoadImageD(keys=keys, reader='NibabelReader', image_only=False),
                EnsureChannelFirstD(keys=keys, channel_dim='no_channel'),
                OrientationD(keys=keys, axcodes="RAS"),
                SpacingD(keys=keys, pixdim=pixdim, mode=mode),
                ThresholdIntensityD(keys=["label"], above=False, threshold=self.class_treshold,
                                    cval=self.class_treshold),
                EnsureTypeD(keys=keys, data_type="tensor"),
                ToDeviceD(keys=keys, device=device)
            ]
        )
