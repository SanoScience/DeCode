import os
import sys
import yaml
import uuid
import argparse

config_dir = os.path.join('config', 'general_config.yaml')
with open(config_dir, 'r') as file:
    config = yaml.safe_load(file)

config_dir2 = os.path.join('config', '3decode_config.yaml')
with open(config_dir2, 'r') as file:
    config2 = yaml.safe_load(file)
config['args'].update(config2['args'])
config['experiment'].update(config2['experiment'])

#read experiment parameters from file     
if config['experiment']['source'] == 'cmd':
    print(" *** Applying configuration from commandline experiment parameters ***")
    parser = argparse.ArgumentParser(
                prog='Conditioning',
                description='Experiments',
                epilog='Hello!')
    
    parser.add_argument('--cuda_device_id', default=0, type=int, choices=[0,1])
    parser.add_argument('--tabular_module', default='FiLM', type=str, choices=['FiLM', 'INSIDE', 'DAFT'])
    parser.add_argument('--is_regression', default=False, action='store_true')
    parser.add_argument('--is_unet_skip', default=False, action='store_true')
    parser.add_argument('--is_inference_regression', default=False, action='store_true')
    #3decode
    parser.add_argument('--experiment', default='size', type=str, choices=['size', 'shape', 'sizesOfShapes', 'varSize', 'varSizeAndShape'])
    parser.add_argument('--features_type', default='onehot', type=str, choices=['onehot', 'radiomics'])
    parser.add_argument('--regression_criterion', default='rmse', type=str, choices=['rmse', 'mse', 'bce'])
    
    args_dict = vars(parser.parse_args())
    print(args_dict)
    print("----------------------------------")
    config['args'].update(args_dict)
 
args = argparse.Namespace(**config['args'])

#if debugger is active will log to comet only if flag debug_comet is  True
if hasattr(sys, 'gettrace') and sys.gettrace() is not None:
    is_debug = True
    if args.debug_comet:
        args.comet = True
    else:
        args.comet = False
else:
    # when debugger off - always logs on comet
    is_debug = False
    args.comet = True
 
import time
import itertools
import random
import json
import glob
import warnings
import numpy as np
import pandas as pd
from comet_ml import Experiment
from natsort import natsorted
from sklearn.model_selection import KFold

if args.comet:
    experiment = Experiment(project_name="3DeCode")
    
    if args.tags:
        tags_list = args.tags.split('#')
        if tags_list:
            experiment.add_tags(tags_list)
            
    experiment.add_tags([args.model_name, f'cuda:{args.cuda_device_id}', str(os.getpid())])
    experiment.log_asset(config_dir)
    experiment.log_code('src/train_3decode.py')
    experiment.log_code('src/data_augmentation_3decode.py')
    if args.model_name == "decode":
        experiment.log_code('src/models/decode.py')
    experiment.log_parameters(vars(args))
    
    unique_name = experiment.get_name()
    if isinstance(args.module_positions, list):
        mod_number = np.array(args.module_positions).sum()
    elif args.module_positions:
        mod_number = 5
    else:
        mod_number = 0
    new_exp_name = f"{unique_name}_{args.loss_name}_m-{args.tabular_module}_e{args.epochs}_b{args.batch_size}_s{args.seed}_embbt{args.bottleneck_dim}_bias-{args.bias}_ds-{args.deep_supervision}_reg{args.is_regression}-{args.features_loss_ratio}_unet{args.is_unet_skip}_reg_infer{args.is_inference_regression}_exp{args.experiment}_cond{args.features_type}" 
    print('Config:', new_exp_name)
    experiment.set_name(new_exp_name)
else:
    from dummy_experiment import DummyExperiment
    experiment = DummyExperiment()
    new_exp_name = "test"
    
    args.batch_size=2
    n = 1
    args.validation_interval = 5*n
    args.log_batch_interval = 5*n
    args.log_metrics_interval = 5*n
    args.log_slice_interval = 5*n
    args.log_3d_scene_interval_training = 5*n
    args.log_3d_scene_interval_validation = 5*n
    args.hausdorff_log_epoch = 50*n
    args.save_interval=5*n
    args.cache_dir = os.path.join(args.cache_dir, 'debug', )   
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, 'debug')   
    unique_name = uuid.uuid4().hex

#SETUP CONFIG
data_preproc_string = f"patch_{('_').join(list([str(i) for i in args.patch_size]))}_class_{args.classes}_cudaid_{args.cuda_device_id}"
args.cache_dir = os.path.join(args.cache_dir, data_preproc_string) 
args.data_radiomics_path = os.path.join(args.data_radiomics_path, args.experiment)
args.data = os.path.join(args.data, args.experiment)
args.checkpoint_dir = os.path.join(args.checkpoint_dir, data_preproc_string)
checkpoint_directory = os.path.join(args.checkpoint_dir, args.model_name)
if not os.path.exists(checkpoint_directory):
    os.makedirs(checkpoint_directory)

if not os.path.exists(args.cache_dir):
    os.makedirs(os.path.join(args.cache_dir, 'train'))
    os.makedirs(os.path.join(args.cache_dir, 'val'))
    os.makedirs(os.path.join(args.cache_dir, 'test'))

if args.clear_cache:
    print("Clearning cache...")
    train_cache = glob.glob(os.path.join(args.cache_dir, 'train/*.pt'))
    val_cache = glob.glob(os.path.join(args.cache_dir, 'val/*.pt'))
    if len(train_cache) != 0:
        for file in train_cache:
            os.remove(file)
    if len(val_cache) != 0:
        for file in val_cache:
            os.remove(file)         
    print(f"Cleared cache in dir: {args.cache_dir}, train: {len(train_cache)} files, val: {len(val_cache)} files.")    
    
if args.clear_test_cache:
    test_cache = glob.glob(os.path.join(args.cache_dir, 'test/*.pt'))
    if len(test_cache) != 0:
        for file in test_cache:
            os.remove(file)
        print(f"Cleared test cache: {len(test_cache)} files.")
        
# TORCH modules
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import SubsetRandomSampler, RandomSampler

# MONAI modules
from monai.networks.nets import VNet, AttentionUnet, UNETR, SwinUNETR

#MODELS
from models.decode import DeCode
from models.resunet import ResUNet

from monai.networks.utils import one_hot
from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss
from torch.nn import MSELoss, L1Loss, BCEWithLogitsLoss

# segmentation metrics 
from monai.metrics import HausdorffDistanceMetric, MeanIoU, DiceMetric
from monai.metrics import CumulativeAverage
from monai.optimizers import WarmupCosineSchedule

# training
from monai.utils import set_determinism
from monai.data import set_track_meta, ThreadDataLoader, decollate_batch, DataLoader
from monai.data.dataset import PersistentDataset
from monai.inferers import sliding_window_inference
from monai.data.utils import worker_init_fn

# local modules
from cuda import setup_cuda
from data_utils.data_viewer import create_fig_3_views
from data_augmentation import Transforms
from tabular_data import RadiomicsDataLoader, get_tabular_config
from logger import Logger

#REPRODUCIBLITY 
if args.seed != -1:
    #monai seed
    set_determinism(seed=args.seed)
    
    seed = args.seed
    NP_MAX = np.iinfo(np.uint32).max
    MAX_SEED = NP_MAX + 1 
    os.environ["PYTHONHASHSEED"] = str(seed)
    seed = int(seed) % MAX_SEED
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # some operations cannot be made deterministic
    if args.deterministic_algorithms:
        os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:8"
        if args.deterministic_debug_mode:
            #full non-deterministic operations warning supression
            torch.set_deterministic_debug_mode('default')
        else:
            torch.use_deterministic_algorithms(mode=args.deterministic_algorithms, warn_only=True)
else:
    g = None
    seed_worker = None
# Average mixed precision settings, default to torch.float32
if args.use_scaler:
    TORCH_DTYPES = {
    'bfloat16': torch.bfloat16,    
    'float16': torch.float16,     
    'float32': torch.float32
    }
    scaler = torch.cuda.amp.GradScaler()
    autocast_d_type=TORCH_DTYPES[args.autocast_dtype]
    if autocast_d_type == torch.bfloat16:
        os.environ["TORCH_CUDNN_V8_API_ENABLED"]="1"
        if torch.cuda.is_bf16_supported():
            print("bfloat16 is supported!")
        else:
            print("bfloat16 is NOT supported, fall back to float32")
            autocast_d_type=torch.float32
        # detect gradient errors - debug cuda C code
    if autocast_d_type != torch.float32:
        torch.autograd.set_detect_anomaly(True)
else:
    scaler = None
    autocast_d_type = torch.float32 

# SETUP CUDA
setup_cuda(use_memory_fraction=args.gpu_frac,
           num_threads=args.num_threads,
           device=args.device,
           visible_devices=args.visible_devices,
           use_cuda_with_id=args.cuda_device_id)
if args.device == 'cuda':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu", index=int(args.cuda_device_id))
    if device.type == "cuda":
        device_name = torch.cuda.get_device_name(int(args.cuda_device_id))
#Precision, benchmark
if args.benchmark_speedup:
    torch.backends.cudnn.benchmark = True
# precision for nn.modules : eg. nn.conv3d - # Nvidia Ampere 
torch.backends.cudnn.allow_tf32 = args.use_tf32
# precision for linear algebra - eg. interpolations and elastic transforms
torch.backends.cuda.matmul.allow_tf32 = args.use_tf32    
    
    
def extract_pid_from_path(path):
    pid = path.split("/")[-2]
    return pid

def save_metrics_all(experiment, fold_results, args):
    df = pd.DataFrame(fold_results)
    df.columns = df.iloc[0]
    df = df[1:]
    mean_row = df.mean()
    std_row = df.std()
    mean_row = pd.DataFrame(mean_row).T
    std_row = pd.DataFrame(std_row).T
    mean_row.index = ['Mean']
    std_row.index = ['Std']
    df = pd.concat([df, mean_row, std_row])
    experiment.log_table(f"all_folds_metrics.csv", df)

    std_row.index = ['Mean']
    mean_row = mean_row.round(4)
    mean_row = (mean_row).astype(str)
    std_row = std_row.round(4)
    std_row = (std_row).astype(str)
    mean_row = mean_row + " +/-" + std_row
    mean_row["Experiment"] = new_exp_name
    mean_row = mean_row.drop(columns=["fold"])

    if args.tabular_data:
        mean_row["module"] = args.tabular_module
        mean_row["features"] = args.tabular_features
    else:
        mean_row["module"] = ""
        mean_row["features"] = ""

    mean_row["model"] = args.model_name
    experiment.log_table(f"all_folds_metrics_small.csv", mean_row)

# TRANSFORMS
trans = Transforms(args, device)
set_track_meta(True)
logger = Logger(num_classes=args.classes, is_log_3d=args.is_log_3d)

# MONAI DATA
nifti_paths_scans = natsorted(glob.glob(os.path.join(args.data, 'scans', '*.nii.gz'), recursive=False))
nifti_paths_labels = natsorted(glob.glob(os.path.join(args.data, 'labels', '*.nii.gz'), recursive=False))
if args.features_type == "onehot":   
    features_file = args.conditioning_file
elif args.features_type == "radiomics":
    features_file = args.shape_features_file
    
if args.tabular_data:
    rdl = RadiomicsDataLoader(args.data_radiomics_path, args.conditional_embeddings, 
                             args.radiomics_shape_features_config, feature_selection=args.feature_selection, file_name=features_file)
    if args.use_random_features:
        x = np.random.standard_normal(size=(len(rdl)-1, rdl.entity_dim))
        shape_features = ((x-np.min(x))/(np.max(x)-np.min(x))).astype(np.float32)
    else:
        shape_features = rdl.get_items(list(range(len(nifti_paths_scans))))
    nifti_list = [{args.keys[0]: scan, args.keys[1]: label, args.keys[2]: radiomics} for (scan, label, radiomics) in zip(nifti_paths_scans, nifti_paths_labels, shape_features)]
else:
    nifti_list = [{args.keys[0]: scan, args.keys[1]: label} for (scan, label) in zip(nifti_paths_scans, nifti_paths_labels)]

train_dataset = PersistentDataset(nifti_list, trans.train_transform, cache_dir=os.path.join(args.cache_dir, 'train'))
val_dataset = PersistentDataset(nifti_list, trans.val_transform, cache_dir=os.path.join(args.cache_dir, 'val'))
if args.perform_test:
    test_dataset = PersistentDataset(nifti_list, trans.val_transform, cache_dir=os.path.join(args.cache_dir, 'test'))


# DATA SPLIT
training_ids = []
validation_ids = []
testing_ids = []
if args.use_json_split:
    json_split_path = os.path.join('config', 'data_split.json')
    with open(json_split_path) as f:
        ds = json.load(f)   
    #shuffle data only when reproducibility seed is set
    if args.seed != -1:
        random.shuffle(ds['training'])
        random.shuffle(ds['validation'])
    train_ids = [[idx for idx, s in enumerate(nifti_paths_scans) if visit_id in s] for visit_id in ds['training']]
    val_ids = [[idx for idx,s in enumerate(nifti_paths_scans) if visit_id in s] for visit_id in ds['validation']]
    train_ids = list(itertools.chain(*train_ids))
    val_ids = list(itertools.chain(*val_ids))
    #Debugging small subset
    if not args.comet and args.small_datset_debug:
        train_ids = train_ids[:int(0.2*len(train_ids))]
        val_ids = train_ids[:int(0.5*len(val_ids))]
    if args.training_data_fraction < 1.0:
        train_ids = train_ids[:int(args.training_data_fraction*len(train_ids))]
        
    training_ids.append(train_ids)
    validation_ids.append(val_ids)
else:
    indices = np.arange(len(nifti_list))
    train_ids, val_ids, test_ids = np.split(indices, [int(
        (1.0 - args.test_split - args.val_split) * len(indices)), int((1.0 - args.test_split) * len(indices))])
    if args.seed != -1:
        random.shuffle(train_ids)
        random.shuffle(val_ids)
        random.shuffle(test_ids)

    # Small subset for faster debugging
    if not args.comet:
        train_ids = train_ids[:int(0.3*len(train_ids))]
        if args.inference_model_path is not None:
            train_ids = train_ids[:4]
        val_ids = train_ids[:int(0.5*len(val_ids))]
    if args.training_data_fraction < 1.0:
        train_ids = train_ids[:int(args.training_data_fraction*len(train_ids))]
    
    training_ids.append(train_ids.tolist())
    validation_ids.append(val_ids.tolist())
    testing_ids.append(test_ids.tolist())

if args.classes<2:
    print("Warning! - cannot use weighted CE, for less than 2 classes segmentation task.")
    args.weighted_ce = False
        
if args.weighted_ce:
    weights = torch.from_numpy(np.load('data/china/class_weights.npy')).to(dtype=torch.float32, device=device)
    weights[0]=args.background_weight
    assert(len(weights) == args.classes)

if args.loss_name == "DiceLoss" and args.classes == 1:
    criterion = DiceLoss(include_background = args.include_background_loss, sigmoid=True)
elif args.loss_name == "DiceCELoss":
    if args.weighted_ce:
        criterion = DiceCELoss(include_background=args.include_background_loss, to_onehot_y=True, softmax=True,
                               lambda_ce=1.0, lambda_dice=1.0, ce_weight=weights)
    else:
        criterion = DiceCELoss(include_background=args.include_background_loss, to_onehot_y=True, softmax=True,
                               lambda_ce=1.0, lambda_dice=1.0)
elif args.loss_name == "DiceFocalLoss" and args.classes == 1:
    criterion = DiceFocalLoss(include_background=args.include_background_loss, sigmoid=True,
                              focal_weight=args.focal_weight, lambda_dice=args.dice_ratio, lambda_focal=args.focal_ratio)
elif args.loss_name == "MAE" and args.classes == 1:
    criterion = L1Loss()
elif (args.loss_name == "MSE" or args.loss_name == "RMSE") and args.classes == 1:
    criterion_mse = MSELoss()

if args.regression_criterion == "bce":
    criterion_regression = BCEWithLogitsLoss()
elif args.regression_criterion == "mse":
    criterion_regression = MSELoss()
elif args.regression_criterion == "rmse":
    criterion_regression = MSELoss()

counter_3d_images = 0
test_standard_devs = None

## TRAINING_STEP
def training_step(batch_idx, train_data, args, model):
    with torch.cuda.amp.autocast(enabled=args.use_scaler, dtype=autocast_d_type, cache_enabled=args.cache_enabled):
        if args.tabular_data:
            radiomics = train_data["radiomics"].to(device)
            if args.deep_supervision:
                output, ds = model(train_data["image"], radiomics)
                loss = criterion(output, train_data["label"].long()) + 0.5 * criterion(ds, train_data["label"].long())
            else:
                output, reg_l2, features_regression, conditioning_stats, _, _  = model(train_data["image"], radiomics)
                segmentation_loss = criterion(output, train_data["label"].long())
                if reg_l2 is not None:
                    regularization_term = args.reg_term * reg_l2
                else:
                    regularization_term = 0
                if features_regression is not None:
                    if args.regression_criterion == "rmse":
                        features_loss = args.features_loss_ratio * torch.sqrt(criterion_regression(torch.sigmoid(features_regression), radiomics))
                    elif args.regression_criterion == "mse":
                        features_loss = args.features_loss_ratio * criterion_regression(torch.sigmoid(features_regression), radiomics)
                    elif args.regression_criterion == "bce":
                        features_loss = args.features_loss_ratio * criterion_regression(features_regression, radiomics)
                        
                else:
                    features_loss = 0
                loss = segmentation_loss + regularization_term + features_loss
        else:
            if args.deep_supervision:
                output, ds = model(train_data["image"])
                loss = criterion(output, train_data["label"].long()) + 0.5 * criterion(ds, train_data["label"].long())
            else:
                output = model(train_data["image"])
                loss = criterion(output, train_data["label"].long())

    if args.use_scaler:
        scaler.scale(loss).backward()
        if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
            if args.grad_clip:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
    else:
        loss.backward()
        if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
            if args.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

    #Prediction
    if args.classes == 1:
        # Binary Classification Task
        pred = trans.post_pred(output).long()
    else:
        # Multiclass Classification Task
        pred = trans.post_pred_train(output).long()

    #Basic console log
    epoch_time = time.time() - start_time_epoch
    batch_size = train_data["image"].shape[0]
    train_loss_cum.append(loss.item(), count=batch_size)
    if args.is_regression:
        feature_loss_cum.append(features_loss.item(), count=batch_size)
    if reg_l2 is not None:
        experiment.log_metric("L2", reg_l2.item())   
        if args.is_log_conditioning:
            metric_keys = [[f"{stage}_{key}" for key in list(conditioning_stats.keys())[1:]] for stage in conditioning_stats['decoder_stage']]
            metric_values  = [[conditioning_stats[key][stage] for key in list(conditioning_stats.keys())[1:]] for stage in range(len(conditioning_stats['decoder_stage']))]
            conditioning_metrics = dict(zip(list(itertools.chain(*metric_keys)), list(itertools.chain(*metric_values))))
            experiment.log_metrics(conditioning_metrics)
    if (batch_idx + 1) % args.log_batch_interval == 0 or (batch_idx + 1) == len(train_loader):
        print(" ", end="")
        print(f"Batch: {batch_idx + 1}/{len(train_loader)} - "
                f" Loss: {train_loss_cum.aggregate():.4f}."
                f" MSE loss: {feature_loss_cum.aggregate():.4f}."
                f" Time: {epoch_time:.2f}s.")
    #METRICS
    #calculate metrics every nth epoch
    if (epoch+1) % args.log_metrics_interval == 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for func in seg_metrics:
                if type(func).__name__ == 'HausdorffDistanceMetric' and (epoch+1) <= args.hausdorff_log_epoch:
                    func.append(torch.inf) 
                    continue
                if args.classes == 1:
                    func(y_pred=pred, y=train_data["label"])
                else:
                    func(y_pred=one_hot(pred, num_classes=args.classes), y=one_hot(train_data["label"], num_classes=args.classes))

        # aggregate on epoch's end
        if (batch_idx + 1) == len(train_loader):
            seg_metric_results = [func.aggregate().mean().item() for func in seg_metrics]

            # log running average for metrics
            train_dice_cum.append(seg_metric_results[0])
            train_jac_cum.append(seg_metric_results[1])
            if (epoch+1) > args.hausdorff_log_epoch:
                train_hd95_cum.append(args.pixdim*seg_metric_results[2])

            print(f" Train metrics:\n"
                f"  * Seg.: dice: {seg_metric_results[0]:.3f}, mIoU: {seg_metric_results[1]:.3f}, hd95: {args.pixdim*seg_metric_results[2]:.3f}.")

    # log visual results to comet.ml
    if args.is_log_image and batch_idx == 2:
        pred_0 = pred[0].squeeze().detach().cpu().numpy()
        label_0 = train_data["label"][0].long().squeeze().detach().cpu().numpy()
        image_0 = train_data["image"][0].squeeze().detach().cpu().numpy()
        image_log_out = create_fig_3_views(image_0, label_0, pred_0)
        experiment.log_image(image_log_out, name=f'train_3v_f_{fold}_e{(epoch + 1):04}_b{batch_idx + 1:02}')

### VALIDATION STEP ###
def validation_step(batch_idx, val_data, args):
    global counter_3d_images
    with torch.cuda.amp.autocast(enabled=args.use_scaler, dtype=autocast_d_type, cache_enabled=args.cache_enabled):
        if args.tabular_data:
            val_output = model(val_data["image"], val_data["radiomics"].to(device))[0]
        else:
            val_output = model(val_data["image"])

        if args.classes == 1:
            val_preds = [trans.post_pred(i).long() for i in decollate_batch(val_output)]
            val_labels = [i for i in decollate_batch(val_data["label"])]
        else:
            val_preds = [trans.post_pred(i).long() for i in decollate_batch(val_output)]
            val_labels = [i for i in decollate_batch(val_data["label"])]

    #METRICS
    #calculate metrics every validation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for func in seg_metrics:
            if type(func).__name__ == 'HausdorffDistanceMetric' and (epoch+1) <= args.hausdorff_log_epoch:
                func.append(np.inf) 
                continue
            if args.classes == 1:
                func(y_pred=val_preds, y=val_labels)
            else:
                func(y_pred=[one_hot(i, args.classes, dim=0) for i in val_preds], y=[one_hot(i, args.classes, dim=0) for i in val_labels])

    if (batch_idx + 1) == len(val_loader):
        seg_metric_results = [func.aggregate().mean().item() for func in seg_metrics]

        # log running average for metrics
        val_dice_cum.append(seg_metric_results[0])
        val_jac_cum.append(seg_metric_results[1])
        if (epoch+1) > args.hausdorff_log_epoch: 
            val_hd95_cum.append(args.pixdim * seg_metric_results[2])

        print(f" Validation metrics:\n"
            f"  * Seg.: dice: {seg_metric_results[0]:.3f}, mIoU: {seg_metric_results[1]:.3f}, hd95: {args.pixdim * seg_metric_results[2]:.3f}.")

    if args.is_log_image and batch_idx == 0:
        pid = val_data["label"][0].meta['filename_or_obj'].split("/")[-1].replace('.nii.gz', '')
        pred_0 = val_preds[0].squeeze().detach().cpu().numpy()
        label_0 = val_labels[0].squeeze().detach().cpu().numpy()
        image_0 = decollate_batch(val_data["image"])[0].squeeze().detach().cpu().numpy()
        image_log_out = create_fig_3_views(image_0, label_0, pred_0)
        experiment.log_image(image_log_out, name=f'val_f{fold}_e{(epoch + 1):04}_b{batch_idx + 1:02}_{pid}_3v')
        if args.is_log_3d and counter_3d_images < args.max_3d_scans and (epoch+1) % args.log_3d_scene_interval_validation == 0:
            if np.sum(pred_0) > 0:
                counter_3d_images += 1
                view_3d = logger.log_3dscene_comp(pred_0, label_0, num_classes=args.classes-1, scene_size=1024)
                experiment.log_image(view_3d, name=f'val_f{fold}_e{(epoch + 1):04}_b{batch_idx + 1:02}_{pid}_3d')


### TEST STEP ###
def test_step(batch_idx, test_data, test_id, args, test_loader):
    global counter_3d_images, test_standard_devs
    with torch.cuda.amp.autocast(enabled=args.use_scaler, dtype=autocast_d_type, cache_enabled=args.cache_enabled):
        if args.test_time_tabular_data and args.tabular_data:
            test_output = model(test_data["image"], test_data["radiomics"].to(device))[0]
        else:
            test_output = model(test_data["image"])

        if args.classes == 1:
            test_preds = [trans.post_pred(i).long() for i in decollate_batch(test_output)]
            test_labels = [i for i in decollate_batch(test_data["label"])]
        else:
            test_preds = [trans.post_pred(i).long() for i in decollate_batch(test_output)]
            test_labels = [i for i in decollate_batch(test_data["label"])]

    #METRICS
    #calculate metrics every validation
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for func in seg_metrics:
            if type(func).__name__ == 'HausdorffDistanceMetric' and (epoch+1) <= args.hausdorff_log_epoch:
                func.append(np.inf) 
                continue
            if args.classes == 1:
                func(y_pred=test_preds, y=test_labels)
            else:
                func(y_pred=[one_hot(i, args.classes, dim=0) for i in test_preds], y=[one_hot(i, args.classes, dim=0) for i in test_labels])
        # if args.classes == 1 and (epoch+1) > args.hausdorff_log_epoch:
        #     hausdorff_metric_test(y_pred=test_preds, y=test_labels)
            
    if (batch_idx + 1) == len(test_loader):
        if args.classes > 1: 
            seg_metric_agg = [func.aggregate() for func in seg_metrics]
        else:
            seg_metric_agg = [func.get_buffer() for func in seg_metrics]
        seg_metric_results = [torch.mean(agg).item() for agg in seg_metric_agg]
        seg_metric_results_std = [torch.std(agg).item() for agg in seg_metric_agg]
        test_standard_devs = [seg_metric_results_std[0], seg_metric_results_std[1], seg_metric_results_std[2]*args.pixdim]

        # log running average for metrics
        test_dice_cum.append(seg_metric_results[0])
        test_jac_cum.append(seg_metric_results[1])
        test_hd95_cum.append(args.pixdim * seg_metric_results[2])
        
        print(f" Testset {test_id} metrics:\n"
              f"  * Seg.: dice: {seg_metric_results[0]:.4f}±{seg_metric_results_std[0]:.4f}, mIoU: {seg_metric_results[1]:.4}±{seg_metric_results_std[1]:.4f}, hd95: {args.pixdim * seg_metric_results[2]:.4f}±{args.pixdim * seg_metric_results_std[2]:.4f}.")
    
    if args.is_log_image and (batch_idx+1) % 5 == 0:
        pid = test_data["label"][0].meta['filename_or_obj'].split("/")[-1].replace('.nii.gz', '')
        pred_0 = test_preds[0].squeeze().detach().cpu().numpy()
        label_0 = test_labels[0].squeeze().detach().cpu().numpy()
        image_0 = decollate_batch(test_data["image"])[0].squeeze().detach().cpu().numpy()
        image_log_out = create_fig_3_views(image_0, label_0, pred_0)
        experiment.log_image(image_log_out, name=f'test_{test_id}_{batch_idx + 1:02}_{pid}_3v')
        if args.is_log_3d and counter_3d_images < args.max_3d_scans and (epoch+1) % args.log_3d_scene_interval_validation == 0:
            if np.sum(pred_0) > 0:
                counter_3d_images += 1
                view_3d = logger.log_3dscene_comp(pred_0, label_0, num_classes=args.classes-1, scene_size=1024)
                experiment.log_image(view_3d, name=f'test_3d_{test_id}_e{(epoch+1):04}_{batch_idx + 1:02}_{pid}')
                
# TESTING        
def testing_procedure(inference_path=None):
    def save_all_results(files_list, dataset_name, is_save=True):
            all_results_columns = ['id', 'dice', 'jaccard', 'hd95']
            df = pd.DataFrame(columns=all_results_columns)
            df['id'] = [i['image'].split("/")[-1].replace('.nii.gz', '') for i in files_list]
            df['dice'] =  seg_metrics[0].get_buffer().cpu().numpy()
            df['jaccard'] =  seg_metrics[1].get_buffer().cpu().numpy()
            df['hd95'] =  seg_metrics[2].get_buffer().cpu().numpy() *  args.pixdim
            # df['hd'] =  hausdorff_metric_test.get_buffer().cpu().numpy() * args.pixdim
            if is_save:
                save_dir = os.path.join(unique_dir, f'{dataset_name}_results.csv')
                print('saved results to: ' + save_dir)
                df.to_csv(save_dir, index=False)
            print(f'\n * Results for {dataset_name} * ')
            print(df)
            
    global model, epoch
    with experiment.test(), torch.no_grad():
        print(f"----------------")
        #check if this is the last testing procedure - to run on the best validation score
        #load model from best validation checkpoint, if training finished, otherwise use current epoch model
        if (epoch+1) == args.epochs:
            if is_debug:
                unique_dir = os.path.join(checkpoint_directory, unique_name)
            else:
                unique_dir = os.path.join(checkpoint_directory, unique_name, new_exp_name)      
            best_val_path = os.path.join(unique_dir, f"{save_path_with_parameters}_current_best_val.pt")
            if os.path.isfile(best_val_path):
                checkpoint_dict = torch.load(best_val_path, map_location=device)
                print(f"FINAL TEST: Testing based on best validation checkpoint with dice score: {checkpoint_dict['model_val_dice']:.4f} from epoch: {checkpoint_dict['epoch']}.")
                model.load_state_dict(checkpoint_dict['model_state_dict'], strict=True)
                model = model.to(device)
            else:
                print(f'Missing file: {best_val_path}!. Using current model after {epoch+1} epochs for test')
        else:
            print("Using current training state of the model to perform the test")
        
        if inference_path is not None:
                checkpoint_dict = torch.load(inference_path, map_location=device)
                print(f"Inference: Testing based on best validation checkpoint with dice score: {checkpoint_dict['model_val_dice']:.4f} from epoch: {checkpoint_dict['epoch']}.")
                model.load_state_dict(checkpoint_dict['model_state_dict'], strict=True)
                model = model.to(device)
                unique_dir = os.path.join(os.path.dirname(inference_path))
        model.eval()  
        _ = [func.reset() for func in seg_metrics]
        _ = [cum.reset() for cum in test_metrics_cms]
        hausdorff_metric_test.reset()

        print("Testing on test subset")
        start_time_test = time.time()
        for batch_idx, test_data in enumerate(test_loader):
            test_step(batch_idx, test_data, 'TEST', args, test_loader)
        test_time = time.time() - start_time_test
        print(f"Testing time: {test_time:.2f}s")
        # log every single patient result for final test with model best on validation
        save_all_results([test_dataset.data[i] for i in test_ids], 'testset', is_save=(epoch+1) == args.epochs)
        
        testset_metrics_agg = [cum.aggregate() for cum in test_metrics_cms]
        print(testset_metrics_agg[0], test_standard_devs[0])
        
        table = [["experiment", "testset", "epoch", "dice_mean", "dice_std", "jac_mean", "jac_std", "hd95_test_mean", "hd95_test_std"]]
        test_results = [[new_exp_name, 'TestSplit', epoch+1, testset_metrics_agg[0], test_standard_devs[0],
                                            testset_metrics_agg[1], test_standard_devs[1],
                                            testset_metrics_agg[2], test_standard_devs[2]]]
        experiment.log_metric("test_datadice", testset_metrics_agg[0], epoch=(epoch+1))
        experiment.log_metric("test_data_dice_std", test_standard_devs[0], epoch=(epoch+1))
        # experiment.log_metric("test_data_HD", hausdorff_metric_test.aggregate().item() * args.pixdim, epoch=(epoch+1))
        table.extend(test_results)
        _ = [func.reset() for func in seg_metrics]
        _ = [cum.reset() for cum in test_metrics_cms]
        hausdorff_metric_test.reset()
        
        experiment.log_table(f"test_metrics.csv", table)


# UNET params
feature_maps = tuple(2 ** i * args.n_features for i in range(0, args.unet_depth))
strides = list((args.unet_depth - 1) * (2,))
tab_dim=0
if args.tabular_data:
    if args.conditional_embeddings == "feature":
        tab_dim = rdl.features_dim
        print(f"using features of embbeding dim: {tab_dim}")
    elif args.conditional_embeddings == "entity":
        tab_dim = rdl.entity_dim
        print(f"using features of embbeding dim: {tab_dim}")
    TABULAR_CONFIG = get_tabular_config(tab_dim, args=args)
else:
    TABULAR_CONFIG = None

### CROSS VALIDATION LOOP ###
best_results = [["fold", "dice_test", "jac_test", "hd95_test", "dice_val", "jac_val", "hd95_val"]]
best_fold_results = [[]]

#TRAINING LOOP
# def main():
print("--------------------")
folds = [[training_ids[i], validation_ids[i], testing_ids[i]] for i in range(args.k_splits)]
for fold, (train_ids, val_ids, test_ids) in enumerate(folds):
    print(f"FOLD {fold}")
    print("-------------------")
    
    test_jac_best_val = 0
    test_dice_best_val = 0
    test_hd_best_val = 0
    save_path_with_parameters = f"model-{args.model_name}_{args.model_config}-class-{args.classes}-module-{args.tabular_module}-fold-{fold}"

    #seeded randomness
    if args.use_random_sampler:
        train_sampler = SubsetRandomSampler(train_ids, generator=None)
        val_sampler = SubsetRandomSampler(val_ids, generator=None)
        test_sampler = SubsetRandomSampler(test_ids, generator=None)
    else:
        #iterable as sampler
        train_sampler = train_ids
        val_sampler = val_ids
        test_sampler = test_ids

    if args.use_thread_loader:
        train_loader = ThreadDataLoader(train_dataset, use_thread_workers=True, buffer_size=1, batch_size=args.batch_size,
                                        sampler=train_sampler)
        val_loader = ThreadDataLoader(val_dataset, use_thread_workers=True, buffer_size=1, batch_size=args.batch_size_val,
                                    sampler=val_sampler)
        test_loader = ThreadDataLoader(test_dataset, use_thread_workers=True, buffer_size=1, batch_size=args.batch_size_val,
                                    sampler=test_sampler)
    else:
        train_loader = DataLoader(train_dataset, num_workers=args.num_workers, shuffle=False, batch_size=args.batch_size, sampler=train_sampler, worker_init_fn=worker_init_fn)
        val_loader = DataLoader(val_dataset, num_workers=args.num_workers, shuffle=False, batch_size=args.batch_size_val, sampler=val_sampler, worker_init_fn=worker_init_fn)
        test_loader = DataLoader(test_dataset, num_workers=args.num_workers, shuffle=False, batch_size=args.batch_size_val, sampler=test_sampler, worker_init_fn=worker_init_fn)
    # MODEL INIT
    #TABULAR READY
    if args.model_name == "DeCode":
        torch.manual_seed(args.seed)
        model = DeCode(spatial_dims=3, in_channels=1, out_channels=args.classes, act='relu', norm='batch', bias=args.bias, tabular_module=args.tabular_module,
                          embedding_size=tab_dim, module_bottleneck_dim=args.bottleneck_dim, is_regression=args.is_regression, is_unet_skip=args.is_unet_skip,
                          is_inference_regression=args.is_inference_regression, is_log_conditioning=args.is_log_conditioning, regression_mlp_expansion=args.regression_mlp_expansion,
                          is_embedding=args.is_embedding, is_inference_embedding=args.is_inference_embedding)
    ## NON TABULAR
    elif args.model_name == "AttUnet":
        model = AttentionUnet(spatial_dims=3, in_channels=1, out_channels=args.classes, channels=feature_maps,
                            strides=strides)
    elif args.model_name == "VNet":
        model = VNet(spatial_dims=3, in_channels=1, out_channels=args.classes, dropout_prob=0, bias=args.bias)
    elif args.model_name == "SwinUNETR":
        model = SwinUNETR(img_size=args.padding_size, in_channels=1, out_channels=args.classes)
    elif args.model_name == "ResUnet18":
        torch.manual_seed(args.seed)
        model = ResUNet(spatial_dims=3, in_channels=1, out_channels=args.classes, act='relu',
                        norm="batch", backbone_name='resnet18', bias=args.bias)
    elif args.model_name == "ResUnet34":
        model = ResUNet(spatial_dims=3, in_channels=1, out_channels=args.classes, act='relu',
                        norm="batch", backbone_name='resnet34', bias=args.bias)
    elif args.model_name == "ResUnet50":
        model = ResUNet(spatial_dims=3, in_channels=1, out_channels=args.classes, act='relu',
                        norm="batch", backbone_name='resnet50', bias=args.bias)

    else:
        raise NotImplementedError(f"There are no implementation of: {args.model_name}")

    if args.parallel:
        model = nn.DataParallel(model).to(device)
    else:
        model = model.to(device)

    # Optimizer
    if args.optimizer == "SGD":
        optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=0.9)
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, amsgrad=args.adam_ams,
                            eps=args.adam_eps)
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, eps=args.adam_eps)
    else:
        raise NotImplementedError(f"There are no implementation of: {args.optimizer}")

    if args.continue_training:
        model.load_state_dict(torch.load(args.trained_model, map_location=device)['model_state_dict'])
        optimizer.load_state_dict(torch.load(args.trained_model, map_location=device)['optimizer_state_dict'])
        args.start_epoch = torch.load(args.trained_model)['epoch']
        print(f'Loaded model, optimizer, starting with epoch: {args.start_epoch}')

    if args.scheduler_name == 'cosine_annealing':
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, verbose=False, eta_min=args.lr_min)
    elif args.scheduler_name == 'warmup_cosine':
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, warmup_multiplier=0.01,
                                        t_total=args.epochs)
    elif args.scheduler_name == 'step_lr':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma)
    elif args.scheduler_name == "late_cosine":
        annealing_start = args.epochs - args.scheduler_step
        if annealing_start < 0:
            raise ValueError (f"Annealing first epoch cannot be negative")
        scheduler = CosineAnnealingLR(optimizer, T_max=args.scheduler_step, eta_min=args.lr_min, verbose=False)
    else:
        scheduler=None

    # METRICS
    reduction = 'mean_batch'
    include_background = True
    seg_metrics = [
        DiceMetric(include_background=include_background, reduction=reduction, ignore_empty=True),
        MeanIoU(include_background=include_background, reduction=reduction, ignore_empty=True),
        HausdorffDistanceMetric(include_background=include_background, distance_metric='euclidean', percentile=95,
                                get_not_nans=False, directed=True, reduction=reduction)
    ]
    hausdorff_metric_test = HausdorffDistanceMetric(include_background=include_background, distance_metric='euclidean', percentile=100,
                                get_not_nans=False, directed=True, reduction=reduction)

    treshold_value = 1e-2
    treshold = nn.Threshold(treshold_value, 0)

    # RUNNING_AVERAGES
    # training loss
    train_loss_cum = CumulativeAverage()
    feature_loss_cum = CumulativeAverage()
    seg_loss_cum = CumulativeAverage()
    training_loss_cms = [train_loss_cum, seg_loss_cum]
    # training metrics
    train_dice_cum = CumulativeAverage()
    train_jac_cum = CumulativeAverage()
    train_hd95_cum = CumulativeAverage()
    training_metrics_cms = [train_dice_cum, train_jac_cum, train_hd95_cum]
    # validation metrics
    val_dice_cum = CumulativeAverage()
    val_jac_cum = CumulativeAverage()
    val_hd95_cum = CumulativeAverage()
    val_metrics_cms = [val_dice_cum, val_jac_cum, val_hd95_cum]
    # test metrics
    test_dice_cum = CumulativeAverage()
    test_jac_cum = CumulativeAverage()
    test_hd95_cum = CumulativeAverage()
    test_metrics_cms = [test_dice_cum, test_jac_cum, test_hd95_cum]

    with experiment.train():
        best_dice_score = 0.0
        best_dice_val_score = 0.0
        best_dice_val_epoch = 0
        accum_iter = args.gradient_accumulation
        for epoch in range(args.start_epoch, args.epochs):
            start_time_epoch = time.time()
            print(f"Starting epoch {epoch + 1}")

            epoch_time = 0.0

            model.train()
            for batch_idx, train_data in enumerate(train_loader):
                training_step(batch_idx, train_data, args, model)

            epoch_time = time.time() - start_time_epoch

            # RESET METRICS after training
            _ = [func.reset() for func in seg_metrics]

            # VALIDATION

            model.eval()
            with torch.no_grad():
                if (epoch + 1) % args.validation_interval == 0 and epoch != 0:
                    print("Starting validation...")
                    start_time_validation = time.time()
                    for batch_idx, val_data in enumerate(val_loader):
                        validation_step(batch_idx, val_data, args)
                    val_time = time.time() - start_time_validation
                    print(f"Validation time: {val_time:.2f}s")

                # RESET METRICS after validation
                _ = [func.reset() for func in seg_metrics]

                # AGGREGATE RUNNING AVERAGES
                train_loss_agg = [cum.aggregate() for cum in training_loss_cms]
                feature_loss_agg = feature_loss_cum.aggregate()
                train_metrics_agg = [cum.aggregate() for cum in training_metrics_cms]
                val_metrics_agg = [cum.aggregate() for cum in val_metrics_cms]
                
                #TEST
                #test only if new best on validation
                if best_dice_val_score < val_metrics_agg[0]:
                    print("New best validation; starting testing...")
                    if (epoch + 1) % args.validation_interval == 0 and epoch != 0:
                        print('Testing...')
                        start_time_test = time.time()
                        for batch_idx, test_data in enumerate(test_loader):
                            test_step(batch_idx, test_data, args.experiment, args, test_loader)
                        test_time = time.time() - start_time_test
                        print(f"Testing time: {test_time:.2f}s")
                test_metrics_agg = [cum.aggregate() for cum in test_metrics_cms]

                # RESET METRICS after testing
                _ = [func.reset() for func in seg_metrics]

                # reset running averages
                _ = [cum.reset() for cum in training_loss_cms]
                _ = [cum.reset() for cum in training_metrics_cms]
                _ = [cum.reset() for cum in val_metrics_cms]
                _ = [cum.reset() for cum in test_metrics_cms]
                
                # LOG METRICS TO COMET
                if scheduler is not None:
                    if args.scheduler_name == "late_cosine":
                        if (epoch+1) > annealing_start:  
                            scheduler.step()
                    else:
                        scheduler.step()
                    experiment.log_metric("lr_rate", scheduler.get_last_lr(), epoch=epoch)
                else:
                    experiment.log_metric("lr_rate", args.lr, epoch=epoch)
                experiment.log_current_epoch(epoch)
                # loss - total
                experiment.log_metric("train_loss", train_loss_agg[0], epoch=epoch)
                experiment.log_metric("feature_mse_loss", feature_loss_agg, epoch=epoch)
                # train metrics
                if (epoch+1) % args.log_metrics_interval == 0:
                    experiment.log_metric("train_dice", train_metrics_agg[0], epoch=epoch)
                    experiment.log_metric("train_jac", train_metrics_agg[1], epoch=epoch)
                    if (epoch+1) > args.hausdorff_log_epoch: 
                        experiment.log_metric("train_hd95", train_metrics_agg[2], epoch=epoch)
                # val metrics
                    if (epoch+1) % args.validation_interval == 0:
                        experiment.log_metric("val_dice", val_metrics_agg[0], epoch=epoch)
                        experiment.log_metric("val_jac", val_metrics_agg[1], epoch=epoch)
                        if (epoch+1) > args.hausdorff_log_epoch: 
                            experiment.log_metric("val_hd95", val_metrics_agg[2], epoch=epoch)
                    
                    experiment.log_metric("test_dice", test_metrics_agg[0], epoch=epoch)
                    experiment.log_metric("test_jac", test_metrics_agg[1], epoch=epoch)
                    experiment.log_metric("test_hd95", test_metrics_agg[2], epoch=epoch)
                    if best_dice_val_score < val_metrics_agg[0]:
                        test_dice_best_val = test_metrics_agg[0]
                        test_jac_best_val = test_metrics_agg[1]
                        test_hd_best_val = test_metrics_agg[2]
                #
                
                # # CHECKPOINTS SAVE
                # save current training best TRAIN model - results can be overwritten
                if best_dice_score < train_metrics_agg[0]:
                    save_path = os.path.join(checkpoint_directory, f"{save_path_with_parameters}_current_best_train.pt")
                    torch.save({
                        'epoch': (epoch+1),
                        'model_state_dict': model.state_dict(),
                        'model_val_dice': train_metrics_agg[0],
                        'model_val_jac': train_metrics_agg[1]
                    }, save_path)
                    best_dice_score = train_metrics_agg[0]
                    print(f"Current best train dice score {best_dice_score:.4f}. Model saved!")

                # save best VALIDATION score
                if best_dice_val_score < val_metrics_agg[0]:
                    if is_debug:
                        # warning - best val checkpoint is overwritten in debug mode
                        unique_dir = os.path.join(checkpoint_directory, unique_name)
                    else:
                        # save every experiment configuration new best checkpoints on validation to the new folder
                        unique_dir = os.path.join(checkpoint_directory, unique_name, new_exp_name)
                    if not os.path.exists(unique_dir):
                        os.makedirs(unique_dir)
                    save_path = os.path.join(unique_dir, f"{save_path_with_parameters}_current_best_val.pt")
                    torch.save({
                        'epoch': (epoch+1),
                        'model_state_dict': model.state_dict(),
                        'model_val_dice': val_metrics_agg[0],
                        'model_val_jac': val_metrics_agg[1]
                    }, save_path)
                    best_dice_val_score = val_metrics_agg[0]
                    best_dice_val_epoch = epoch+1
                    print(f"Current best validation dice score {best_dice_val_score:.4f}. Model saved!")

                    # table = [["fold", "dice_test", "jac_test", "hd95_test", "dice_val", "jac_val", "hd95_val"]]
                    table = [["epoch", "dice_val", "jac_val", "hd95_val"]]
                    best_fold_results = [[epoch+1, val_metrics_agg[0], val_metrics_agg[1], val_metrics_agg[2]]]
                    table.extend(best_fold_results)
                    experiment.log_table(f"best_val_metrics_fold_{fold}.csv", table)
                
                #save based on INTERVAL every n epochs even if result on validation is not best
                #unique directory - file wont be overwritten
                if (epoch+1) % args.save_interval == 0 and epoch != 0:
                    if is_debug:
                        unique_dir = os.path.join(checkpoint_directory, unique_name)
                    else:
                        unique_dir = os.path.join(checkpoint_directory, unique_name)
                        
                    path_stats = f"val_{val_metrics_agg[0]:.4f}_train_{train_metrics_agg[0]:.4f}_epoch_{(epoch+1):04}"
                    if not os.path.exists(unique_dir):
                            os.makedirs(unique_dir)
                    #every n epochs save OPTIMIZER and SCHEDULER to allow for continued training
                    if args.save_optimizer and (epoch+1) % args.save_optimiser_interval == 0 and epoch != 0:
                        save_path = os.path.join(unique_dir, f"{save_path_with_parameters}_{path_stats}_state_dicts_optim.pt")
                        torch.save({
                            'epoch': (epoch),
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'lr_scheduler' : scheduler.state_dict(),
                            'model_train_dice': train_metrics_agg[0],
                            'model_train_jac': train_metrics_agg[1],
                            'model_val_dice': val_metrics_agg[0],
                            'model_val_jac': val_metrics_agg[1]
                            }, save_path)
                        print(f"Sate dicts saved! - models train_dice: {train_metrics_agg[0]:.4f}, models val_dice: {val_metrics_agg[0]:.4f}, current best_val_dice: {best_dice_val_score:.4f}..")
                    else:
                        save_path = os.path.join(unique_dir, f"{save_path_with_parameters}_{path_stats}_model_state.pt")
                        torch.save({
                            'epoch': (epoch),
                            'model_state_dict': model.state_dict(),
                            'model_train_dice': train_metrics_agg[0],
                            'model_train_jac': train_metrics_agg[1],
                            'model_val_dice': val_metrics_agg[0],
                            'model_val_jac': val_metrics_agg[1]
                            }, save_path)
                        print(f"Interval model saved! - models train_dice: {train_metrics_agg[0]:.4f}, models val_dice: {val_metrics_agg[0]:.4f}, current best_val_dice: {best_dice_val_score:.4f}.")
                print(f"Epoch: {epoch + 1} finished. Total training loss: {train_loss_agg[0]:.4f} - total epoch time: {epoch_time:.2f}s.")
                
                if args.inference_model_path is not None:
                    if (epoch + 1) in [int(args.epochs * i) for i in [0.2, 0.4, 0.6, 0.8, 1]]:
                        print("Starting testing procedure...")
                        testing_procedure(args.inference_model_path)
                        
    best_results.extend(best_fold_results)
    print(f"Fold {fold} finished!")
    

print(f"Training finished!")
with experiment.train():
    save_metrics_all(experiment, best_results, args)
