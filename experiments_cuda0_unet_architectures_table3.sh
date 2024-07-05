#!/bin/bash

declare -a arr=(
    #UNext
    "python -u src/train.py --cuda_device_id 0 --model_name UNeXt3D --is_unet_skip > logs/arch_unetx.log 2>&1 &"
    #Attention UNet
    "python -u src/train.py --cuda_device_id 0 --model_name AttUNet --is_unet_skip > logs/arch_unet_att.log 2>&1 &"
    #UNet Ronnenberger et al.
    "python -u src/train.py --cuda_device_id 0 --model_name UNetRon --is_unet_skip > logs/arch_unet_ronnenberger.log 2>&1 &"
    #VNet
    "python -u src/train.py --cuda_device_id 0 --model_name VNet --is_unet_skip > logs/arch_unet_v.log 2>&1 &"
    #ResUNet34
    "python -u src/train.py --cuda_device_id 0 --model_name ResUNet34 --is_unet_skip > logs/arch_unet_res34.log 2>&1 &"
)
for i in "${arr[@]}"; do
    echo running experiment;
    {
    eval "$i"
    echo $! > running_exp_cuda0.pid 
    echo  $i 
    cat running_exp_cuda0.pid
    }
    wait;
    echo experiment finished;
done