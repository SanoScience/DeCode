#!/bin/bash

declare -a arr=(
    # # #1. UNet
    "python -u src/train.py --epochs 400 --cuda_device_id 0 --is_unet_skip > logs/unet.log 2>&1 &"
    # # #2. UNet + regression task
    "python -u src/train.py --epochs 400 --cuda_device_id 0 --tabular_data --is_regression --is_unet_skip > logs/unet_reg.log 2>&1 &"
    # # #3. UNet + FiLM - random features
    "python -u src/train.py --epochs 400 --cuda_device_id 0 --tabular_data --tabular_module FiLM --use_random_features --is_unet_skip > logs/unet_film_random.log 2>&1 &"
    # #4. UNet + FiLM (CSF)
    "python -u src/train.py --epochs 400 --cuda_device_id 0 --tabular_data --tabular_module FiLM --is_unet_skip > logs/unet_film.log 2>&1 &"
    #5. UNet + INSIDE (CSF)
    "python -u src/train.py --epochs 400 --cuda_device_id 0 --tabular_data --tabular_module INSIDE --is_unet_skip > logs/unet_inside.log 2>&1 &"
    #6. UNet + DAFT (CSF)
    "python -u src/train.py --epochs 400 --cuda_device_id 0 --tabular_data --tabular_module DAFT --is_unet_skip > logs/unet_daft.log 2>&1 &"
    # #7. UNet + FiLM + regression task (CSF)
    "python -u src/train.py --epochs 400 --cuda_device_id 0 --tabular_data --tabular_module FiLM --is_regression --is_unet_skip> logs/unet_film_regression.log 2>&1 &"
    # #8. UNet + DeCode Embedding + regression task + inference time embedding (LESF)
    "python -u src/train.py --epochs 400 --cuda_device_id 0 --tabular_data --tabular_module FiLM --is_embedding --is_regression --is_unet_skip --is_inference_embedding > logs/unet_film_embedding_inference.log 2>&1 &"
)
for i in "${arr[@]}"; do
    echo running experiment;
    {
    eval "$i"
    echo $! > running_exp_cuda0.pid
    echo  $i sss
    cat running_exp_cuda0.pid
    }
    wait;
    echo experiment finished;
done