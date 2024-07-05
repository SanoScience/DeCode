#!/bin/bash

declare -a arr=(
    #size, radiomics 
    "python -u src/train_3decode.py --cuda_device_id 0 --experiment size --features_type radiomics --is_unet_skip --is_regression > logs/size_radiomics.log 2>&1 &"
    #shape, radiomics
    "python -u src/train_3decode.py --cuda_device_id 0 --experiment shape --features_type radiomics --is_unet_skip > logs/shape_radiomics.log 2>&1 &"
    #sizesOfShapes, radiomics
    "python -u src/train_3decode.py --cuda_device_id 0 --experiment sizesOfShapes --features_type radiomics --is_unet_skip > logs/sofs_radiomics.log 2>&1 &"
    #varSize, radiomics
    "python -u src/train_3decode.py --cuda_device_id 0 --experiment varSize --features_type radiomics --is_unet_skip > logs/varsize_radiomics.log 2>&1 &"
    #varSizesOfShapes, radiomics
    "python -u src/train_3decode.py --cuda_device_id 0 --experiment varSizeAndShape --features_type radiomics --is_unet_skip > logs/varsofs_radiomics.log 2>&1 &"
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