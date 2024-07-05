# Let Me DeCode You: Decoder Conditioning with Tabular Data
This is the official code for "Let Me DeCode You: Decoder Conditioning with Tabular Data" accepted at the 27th International Conference on Medical Image Computing and Computer Assisted Intervention (MICCAI) 2024.

## Overview
![Figure 1. Method overview](figures/DeCode_overview.png?raw=true "DeCodeOverview")

## Abstract
Training deep neural networks for 3D segmentation tasks can be challenging, often requiring efficient and effective strategies to improve model performance. In this study, we introduce a novel approach, DeCode, that utilizes label-derived features for model conditioning to support the decoder in the reconstruction process dynamically, aiming to enhance the efficiency of the training process. DeCode focuses on improving 3D segmentation performance through the incorporation of conditioning embedding with learned numerical representation of 3D-label shape features. Specifically, we develop an approach, where conditioning is applied during the training phase to guide the network toward robust segmentation. When labels are not available during inference, our model infers the necessary conditioning embedding directly from the input data, thanks to a feed-forward network learned during the training phase. This approach is tested using synthetic data and cone-beam computed tomography (CBCT) images of teeth. For CBCT, three datasets are used: one publicly available and two in-house. Our results show that DeCode significantly outperforms traditional, unconditioned models in terms of generalization to unseen data, achieving higher accuracy at a reduced computational cost. This work represents the first of its kind to explore conditioning strategies in 3D data segmentation, offering a novel and more efficient method for leveraging annotated data.

## 3DeCode Synthethic Dataset
![3DeCode dataset examples](figures/3decode.png?raw=true "3DeCode")
We present a novel dataset 3DeCode inspired by CLEVR-Seg [1], extending it to 3D and generating segmentation masks based on conditioning scenario tasks. We design tasks that require conditioning based on Shape, Size, or Shapes of different Sizes (referred to as Mixed). The Varying Mixed segmentation task consists of shapes varying in size and shape, where, e.g., the base spherical shape can result in an ellipsoid and a cube in a cuboid.

## Normalized mean shape features calculated with PyRadiomics on training CBCT dataset [2] 
![Radiomics features](figures/normalised_mean_all_features_plasma_china.png?raw=true "Radiomics features")
Each shape feature is calculated for every tooth (32) separately revealing morphological differences between tooth types. We consider features such as e.g. sphericity, volume, and elongation. These morphometric descriptors analyze size, form, and shape, and are thus closely linked to the morphology of the segmented objects.

## Reproducibility
1. To install dependencies:
```
 conda env create -f environment_decode.yml 
```
2. Training CBCT dataset source: https://www.nature.com/articles/s41467-022-29637-2
3. Dataset split IDs: ```config/data_split.json```
4. 3DeCode dataset - to generate syntethic dataset for all conditioning tasks run:
```
python src/data_utils/3decode_dataset.py
```
5. All necessary variables are stored in configuration files.
```
config/general_config.yaml
config/3decode_config.yaml
```
6. To calculate shape features from labels run:
```python src/data_utils/shape_radiomics.py```
7. To train CBCT segmentation model with default parameters run (in config file yaml one can change parameters source: cmd parser or manual yaml file):
```python src/train.py```
8. To train 3DeCode synthetic data conditioning experiment run:
```python src/train_3decode.py```
9. To reproduce use training shell scripts with corresponding table index in name. To use provided shell scripts first make them executable eg.: ```chmod +x ./experiments_cuda0_cbct_table2.sh```
10. Proposed solution DeCode (table 2 - config. 8) - download PyTorch trained model state dictionary: https://drive.google.com/file/d/1G195nn5f5eyQR9fZzXweNtpGBkJdX-QL/view?usp=sharing
11. Sample 3D segmentation results from the test set using DeCode - NIfTI files: https://drive.google.com/drive/folders/1UW_eFabFdc8xm0_68mO5jxy0UqrvPfbg?usp=sharing

## References
1. Jacenków, Grzegorz, et al. "INSIDE: steering spatial attention with non-imaging information in CNNs." Medical Image Computing and Computer Assisted Intervention–MICCAI 2020: 23rd International Conference, Lima, Peru, October 4–8, 2020, Proceedings, Part IV 23. Springer International Publishing, 2020.
2. Cui, Zhiming, et al. "A fully automatic AI system for tooth and alveolar bone segmentation from cone-beam CT images." Nature communications 13.1 (2022): 2096.
3. Perez, Ethan, et al. "Film: Visual reasoning with a general conditioning layer." Proceedings of the AAAI conference on artificial intelligence. Vol. 32. No. 1. 2018.
4. Fedorov, Andriy, et al. "3D Slicer as an image computing platform for the Quantitative Imaging Network." Magnetic resonance imaging 30.9 (2012): 1323-1341.
5. Sullivan, C., and Alexander Kaszynski. "PyVista: 3D plotting and mesh analysis through a streamlined interface for the Visualization Toolkit (VTK)." Journal of Open Source Software 4.37 (2019): 1450.