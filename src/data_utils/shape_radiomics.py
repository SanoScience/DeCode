import os
import sys
import yaml
import glob
import numpy as np
import pandas as pd
import nibabel as nib
from natsort import natsorted
from tqdm.auto import tqdm
from argparse import Namespace
import radiomics
import SimpleITK as sitk
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.getcwd(), 'src'))
from data_augmentation import Transforms
yaml.Dumper.ignore_aliases = lambda *args : True
print(radiomics.generalinfo.GeneralInfo().getGeneralInfo())

# input_dir = 'data/test/A'
# output_dir = 'data/radiomics/test/A'
# input_dir = 'data/test/B'
# output_dir = 'data/radiomics/test/B'
input_dir = 'data/china'
output_dir = 'data/radiomics/china'
save_results = True
data_count_limit = 97

def main():
    #DATA SETUP
    transform = Transforms(keys=["label"], classes=33)
    nifti_paths_scans = natsorted(glob.glob(os.path.join(input_dir, 'scans', '*.nii.gz'), recursive=False))
    if "china" in input_dir:
        nifti_paths_labels = natsorted(glob.glob(os.path.join(input_dir, 'labels_american', '*.nii.gz'), recursive=False))
    else:
        nifti_paths_labels = natsorted(glob.glob(os.path.join(input_dir, 'labels', '*.nii.gz'), recursive=False))
    nifti_labels_list = [{"label": i} for i in nifti_paths_labels]
    
    # EXCTRACTOR SETUP
    params = os.path.join('config', 'radiomics_params.yaml')
    extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(params)
    print(f"Enabled image types: {extractor.enabledImagetypes}")
    print(f"Enabled features: {extractor.enabledFeatures}")
    feature_names = extractor.enabledFeatures['shape']
    
    # EXTRACTION LOOP
    margin = 1
    radiomics_shape_features = []
    missing_tooth_features = [0.0 for i in range(len(extractor.enabledFeatures['shape']))]
    radiomics.setVerbosity(60)

    for _, label_path in enumerate(tqdm(nifti_labels_list[:data_count_limit])):
        array = transform.none_transform(label_path)
        label_array = array["label"].squeeze().numpy()
        labels = np.unique(label_array)[1:] #discard background class
        mask_features = []
        # for idx in tqdm(range(1,33), leave=False):
        for idx in range(1,33):
            label_features = []
            if idx in labels:
                mask_array = (label_array==idx).astype(np.uint32)
                argmin = np.argwhere(mask_array).min(axis=0)
                argmax = np.argwhere(mask_array).max(axis=0)
                bbox = argmin.tolist() + argmax.tolist()
                x_min =  bbox[0]-margin if bbox[0]-margin>=0 else bbox[0]
                y_min =  bbox[1]-margin if bbox[1]-margin>=0 else bbox[1]
                z_min =  bbox[2]-margin if bbox[2]-margin>=0 else bbox[2]
                x_max =  bbox[3]+margin if bbox[3]+margin<label_array.shape[0] else bbox[3]
                y_max =  bbox[4]+margin if bbox[4]+margin<label_array.shape[1] else bbox[4]
                z_max =  bbox[5]+margin if bbox[5]+margin<<label_array.shape[2] else bbox[5]
                masked_object  = mask_array[x_min:x_max, y_min:y_max, z_min:z_max].astype(np.uint32)
                mask = sitk.GetImageFromArray(masked_object)
                result = extractor.execute(mask, mask)
                for key, v in result.items():
                    if 'original_shape' in key:
                        if isinstance(v, np.ndarray):
                            v = float(v)
                        label_features.append(v)
                mask_features.append(label_features)
            else:
                mask_features.append(missing_tooth_features)
        radiomics_shape_features.append(mask_features)

    #PREPROCESS FEATURES
    visit_ids = [i.split('/')[-1].replace('.nii.gz','').replace('_snapnew','') for i in nifti_paths_labels[:data_count_limit]]
    tooth_ids = list(range(1,33))
    feature_names = extractor.enabledFeatures['shape']
    
    #MAX MIN
    max_feature_values = {key: -1e32 for key in extractor.enabledFeatures['shape']}
    min_feature_values = {key: 1e32 for key in extractor.enabledFeatures['shape']}
    radiomics_feature_dict = {}
    for visit_key, visit in zip(visit_ids, radiomics_shape_features):
        visit_dict={}
        for tooth_key, tooth_features in zip(tooth_ids, visit):
            tooth_features = [float(i) for i in tooth_features]
            tooth_features_dict = dict(zip(feature_names, tooth_features))
            if not all(np.array(tooth_features)==0):
                for feature_key, feature_value in zip(feature_names, tooth_features):
                    if max_feature_values[feature_key]<feature_value:
                        max_feature_values[feature_key] = float(feature_value)
                    if min_feature_values[feature_key]>feature_value:
                        min_feature_values[feature_key] = float(feature_value)
            visit_dict[tooth_key]=tooth_features_dict
        radiomics_feature_dict[visit_key]=visit_dict
    
    #SAVE OUTPUT
    if save_results:
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            os.makedirs(os.path.join(output_dir,'visits'))
            os.makedirs(os.path.join(output_dir,'visits_norm'))
        with open(os.path.join(output_dir, 'radiomics_features.yaml'), 'w') as outfile:
            yaml.dump(radiomics_feature_dict, outfile, default_flow_style=False)
        with open(os.path.join(output_dir, 'radiomics_features_max.yaml'), 'w') as outfile:
            yaml.dump(max_feature_values, outfile, default_flow_style=False)
        with open(os.path.join(output_dir, 'radiomics_features_min.yaml'), 'w') as outfile:
            yaml.dump(min_feature_values, outfile, default_flow_style=False)
        
        #normalise features to <0;1>
        radiomics_features_np = np.array(radiomics_shape_features)
        np.save(os.path.join(output_dir, 'radiomics_features.npy'), radiomics_features_np)
        max_np = np.array(list(max_feature_values.values()))
        min_np = np.array(list(min_feature_values.values()))
        # radiomics_features_norm_np = radiomics_features_np / max_np
        radiomics_features_norm_np = (radiomics_features_np - min_np) / (max_np - min_np)
        #missing tooth have nagative values because of normalisation, encode missing to calcualte mean
        radiomics_features_norm_np[radiomics_features_norm_np<0]=-1
        #mask missing teeth - zeroes - mean from normalised data
        masked = np.ma.masked_equal(radiomics_features_norm_np, -1)
        mean_features_per_tooth = masked.mean(axis=0).data
        #replace missing tooth with zeros
        radiomics_features_norm_np[radiomics_features_norm_np<0]=0
        np.save(os.path.join(output_dir, 'radiomics_features_norm.npy'), radiomics_features_norm_np)

        feature_names[10] = 'Max2DDiameterRow'
        feature_names[9] = 'Max2DDiameterColumn'
        feature_names[8] = 'Max2DDiameterSlice'
        for cmap in ['jet', 'plasma', 'viridis', 'magma']:
            fig, ax = plt.subplots(1,1) 
            fig.set_dpi(450)
            plot = ax.imshow(mean_features_per_tooth.transpose(), cmap=cmap)
            # ax.set_xlabel('Tooth IDs')
            # ax.set_ylabel('Shape features')
            y_ticks_labels = feature_names
            ax.set_xticklabels(list(range(1,33)))
            ax.set_xticks(list(range(0,32)))
            ax.tick_params(axis='x', which='major', labelsize=7)
            ax.set_yticks(list(range(mean_features_per_tooth.shape[1])))
            ax.set_yticklabels(y_ticks_labels, rotation='horizontal', fontsize=8)
            cbar = plt.colorbar(plot, ax=ax, fraction=0.0231, pad=0.05)
            plt.savefig(os.path.join(output_dir, 'plot', f'normalised_mean_all_features_{cmap}.png'), dpi=300, bbox_inches='tight')
            plt.savefig(os.path.join(output_dir, 'plot', f'normalised_mean_all_features_{cmap}.eps'), dpi=300, bbox_inches='tight')

if __name__ == "__main__":
    main()
