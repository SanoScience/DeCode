import numpy as np
from time import sleep
import cv2


def _pad_imgs(img_main, img_ax0, img_ax1, max_shape, input_img_shape):
    pad_ax00, pad_ax01 = (max_shape - input_img_shape[1]) // 2, (max_shape - input_img_shape[2]) // 2
    pad_ax10, pad_ax11 = (max_shape - input_img_shape[0]) // 2, (max_shape - input_img_shape[2]) // 2
    pad_main0, pad_main1 = (max_shape - input_img_shape[0]) // 2, (max_shape - input_img_shape[1]) // 2
    plus_pixel_ax00 = (max_shape - input_img_shape[1]) % 2
    plus_pixel_ax01 = (max_shape - input_img_shape[2]) % 2
    plus_pixel_ax10 = (max_shape - input_img_shape[0]) % 2
    plus_pixel_ax11 = (max_shape - input_img_shape[2]) % 2
    plus_pixel_main0 = (max_shape - input_img_shape[0]) % 2
    plus_pixel_main1 = (max_shape - input_img_shape[1]) % 2
    img_ax0 = np.pad(img_ax0, [(pad_ax00, pad_ax00 + plus_pixel_ax00), (pad_ax01, pad_ax01 + plus_pixel_ax01)])
    img_ax1 = np.pad(img_ax1, [(pad_ax10, pad_ax10 + plus_pixel_ax10), (pad_ax11, pad_ax11 + plus_pixel_ax11)])
    img_main = np.pad(img_main, [(pad_main0, pad_main0 + plus_pixel_main0), (pad_main1, pad_main1 + plus_pixel_main1)])
    return img_main, img_ax0, img_ax1


def _cut_3_views(img, ax0_cut, ax1_cut, ax2_cut, input_img_shape):
    img_main = img[:, :, ax2_cut]
    img_ax0 = img[ax0_cut, :, :]
    img_ax0 = np.repeat(img_ax0, 2, axis=1)
    img_ax1 = img[:, ax1_cut, :]
    img_ax1 = np.repeat(img_ax1, 2, axis=1)
    max_shape = max(img.shape)
    img_main, img_ax0, img_ax1 = _pad_imgs(img_main, img_ax0, img_ax1, max_shape, input_img_shape)
    return img_main, img_ax0, img_ax1


def create_3_views(img, label, pred=None):
    input_img_shape = img.shape
    label_ge_0 = np.where(label > 0)
    ax0_cut, ax1_cut, ax2_cut = int(np.median(label_ge_0[0])), int(np.median(label_ge_0[1])), int(
        np.median(label_ge_0[2]))
    img_main, img_ax0, img_ax1 = _cut_3_views(img, ax0_cut, ax1_cut, ax2_cut, input_img_shape)
    label_main, label_ax0, label_ax1 = _cut_3_views(label, ax0_cut, ax1_cut, ax2_cut, input_img_shape)
    img3v = np.concatenate([img_main, img_ax0, img_ax1], axis=1)
    label3v = np.concatenate([label_main, label_ax0, label_ax1], axis=1)
    pred3v = None
    if pred is not None:
        pred_main, pred_ax0, pred_ax1 = _cut_3_views(pred, ax0_cut, ax1_cut, ax2_cut, input_img_shape)
        pred3v = np.concatenate([pred_main, pred_ax0, pred_ax1], axis=1)
    return img3v, label3v, pred3v


def add_masks_to_view(img3v, label3v, pred3v=None):
    img3v = cv2.cvtColor(img3v, cv2.COLOR_GRAY2BGR)
    contours, _ = cv2.findContours(label3v.astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    img3v = cv2.drawContours(img3v, contours, -1, (0, 255, 0), 1)

    if pred3v is not None:
        contours, _ = cv2.findContours(pred3v.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        img3v = cv2.drawContours(img3v, contours, -1, (0, 0, 255), 1)
    return img3v.astype(np.int32)


def create_fig_3_views(img, label, pred=None, plot=False):
    img = img*255
    img3v, label3v, pred3v = create_3_views(img, label, pred)
    img3v = add_masks_to_view(img3v, label3v, pred3v)
    if plot:
        fig, ax = plt.subplots()
        ax.axis("off")
        ax.imshow(img3v, interpolation=None)
        plt.show()
    return img3v


if __name__ == '__main__':
    import sys

    sys.path.append("..")
    from data_augmentation import Transforms
    from monai.data import Dataset, DataLoader
    from arg_parser import args
    from natsort import natsorted
    import glob
    import os
    import matplotlib.pyplot as plt
    from skimage.measure import find_contours

    data_root_dir = "../../../../data/nifti_bssfp_refined/"
    nifti_paths_scans = natsorted(glob.glob(os.path.join(data_root_dir, '**', '*mri.nii.gz'), recursive=True))
    nifti_paths_labels = natsorted(
        glob.glob(os.path.join(data_root_dir, '**', '*label.nii.gz'), recursive=True))
    nifti_list = [{args.keys[0]: scan, args.keys[1]: label} for (scan, label) in
                  zip(nifti_paths_scans, nifti_paths_labels)]
    trans = Transforms(args)
    train_dataset = Dataset(nifti_list, trans.train_transform)

    train_loader = DataLoader(train_dataset, batch_size=1)
    min_data = np.inf
    min_label = np.inf
    max_data = -np.inf
    max_label = -np.inf

    for data in train_loader:
        img = data['image']
        label = data['label']
        img = img.squeeze().numpy()
        label = label.squeeze().numpy()

        min_data = min(min_data, np.min(img))
        min_label = min(min_label, np.min(label))
        max_data = max(max_data, np.max(img))
        max_label = max(max_label, np.max(label))
        print(f"Min img: {min_data}, max img {max_data}")
        print(f"Min label: {min_label}, max label {max_label}")
        create_fig_3_views(img, label, plot=True)

    print(f"Min img: {min_data}, max img {max_data}")
    print(f"Min label: {min_label}, max label {max_label}")
