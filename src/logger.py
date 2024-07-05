from typing import Optional, Union
import pandas as pd
import numpy as np
import torch
import itertools
import subprocess
import glob
import os
import h5py
import cv2
import io
import pyvista as pv
from matplotlib import colors
from matplotlib import pyplot as plt
from matplotlib.pylab import cm

pv.global_theme.font.size = 26
pv.global_theme.font.label_size = 22
pv.global_theme.font.color = 'black'


def start_xvfb(display: int = 98, is_jupyter: bool = False):
    print("Starting pyvista xvfb server")
    xvfb = subprocess.check_output("ps -ef | grep Xvfb | grep screen", shell=True)
    display = f':{display}'
    if display in str(xvfb):
        os.environ['DISPLAY'] = display
        print(f"Xvfb process was working, using DISPLAY={display})")
    else:
        pv.start_xvfb()
        print(f"Xvfb started, using DISPLAY={display}")
    if is_jupyter:
        pv.set_jupyter_backend('panel')


class Logger():
    def __init__(self,
                 num_classes: int = -1,
                 is_log_3d: bool = False,
                 camera_views: list[int] = [3, 5, 6, 7],
                 use_slicer_colors: bool = True) -> None:

        self.classes_num = num_classes
        camera_positions = list(map(list, itertools.product([-1, 1], repeat=3)))
        self.camera_positions = [camera_positions[i] for i in camera_views]
        self.camera_positions_LR = [[1, 0, 0], [-1, 0, 0]]
        self.camera_positions_AP = [[0, 1, 0], [0, -1, 0]]

        if is_log_3d:
            print("Starting pyvista xvfb server")
            xvfb = subprocess.check_output("ps -ef | grep Xvfb | grep screen", shell=True)
            if ':99' in str(xvfb):
                os.environ['DISPLAY'] = ':99'
                print("Xvfb process was working, using DISPLAY=:99")
            else:
                pv.start_xvfb()
                print("Xvfb started, using DISPLAY=:99")
            pv.set_jupyter_backend('panel')

        if use_slicer_colors:
            tooth_colors = pd.read_csv('src/misc/slicer_33_colormap.txt', delimiter=" ", header=None)
            tooth_colors_df = tooth_colors.iloc[:, 2:5]
            tooth_colors_df.columns = ['r', 'g', 'b']
            colorspace = tooth_colors_df.to_numpy() / 255

            if self.classes_num == -1:
                self.color_map = colorspace
            else:
                self.color_map = colorspace[:(self.classes_num + 1)]
            self.listed_color_map = colors.ListedColormap(self.color_map, 'slicer_colors')

    def pad_to_square(self, a, pad_value=0):
        new_shape = np.array(3 * [max(a.shape)])
        padded = pad_value * np.ones(new_shape, dtype=a.dtype)
        # trivial padding - without centering
        padded[:a.shape[0], :a.shape[1], :a.shape[2]] = a
        return padded

    def symmetric_padding_3d(self, array, target_shape, pad_value=0):
        if len(array.shape) != 3:
            raise ValueError("Input array must be 3-dimensional.")

        if all(array.shape == target_shape):
            return array.copy()  # No padding required, return a copy of the original array

        padded_array = np.full(target_shape, pad_value, dtype=array.dtype)
        pad_widths = []
        for dim in range(3):
            pad_total = target_shape[dim] - array.shape[dim]
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            pad_widths.append((pad_left, pad_right))

        padded_array[
        pad_widths[0][0]:target_shape[0] - pad_widths[0][1],
        pad_widths[1][0]:target_shape[1] - pad_widths[1][1],
        pad_widths[2][0]:target_shape[2] - pad_widths[2][1]
        ] = array

        return padded_array

    def log_3dscene_comp(self, volume: np.array, volume_gt: np.array, num_classes: int = -1, scene_size: int = 480,
                         camera_pos: list = [0.5, -1, 0.5]) -> np.array:

        scene_size = (scene_size,) * 2
        labels = dict(xlabel='R', ylabel='P', zlabel='S')

        if num_classes == 0:
            num_classes = 1
        elif num_classes == -1:
            num_classes = self.classes_num - 1

        data = pv.UniformGrid()
        data_gt = pv.UniformGrid()

        # prediction
        data.dimensions = np.array(volume.shape) + 1
        data.cell_data['values'] = volume.ravel(order='F')
        tresh_data = data.threshold(1, scalars='values')

        # ground_truth
        data_gt.dimensions = np.array(volume_gt.shape) + 1
        data_gt.cell_data['values'] = volume_gt.ravel(order='F')
        tresh_data_gt = data_gt.threshold(1, scalars='values')

        # plotter
        p = pv.Plotter(window_size=scene_size, off_screen=True, lighting='three lights')
        p.set_background('#c1c3e8', top='#7579be')
        p.add_axes(line_width=6, ambient=0.5, **labels)

        sargs = dict(
            title='decoder_conditioning',
            title_font_size=16,
            label_font_size=12,
            shadow=True,
            n_labels=self.classes_num,
            italic=False,
            fmt="%.0f",
            font_family="arial",
        )

        # PLOT SCENES
        # prediction
        pred = p.add_mesh(tresh_data, cmap=self.listed_color_map, scalars="values", clim=[-0.5, num_classes + 0.5],
                          scalar_bar_args=sargs, smooth_shading=False)
        #canvas bounding box
        # cube = pv.Cube()
        # cube_outline = cube.outline(generate_faces = True).points*2
        # outline = data.outline()
        # bounds=np.array(list(outline.bounds[1::2]))
        # ratio=0
        # outline.points = bounds/2 * (1 + cube_outline*(1-2*ratio))
        # cube.points = bounds/2 * (1 + (cube.points*2) *(1-2*ratio))
        # pred_box = p.add_mesh(outline, color="b")

        p.camera_position = camera_pos
        pred_scene_PA = p.screenshot(return_img=True)
        p.camera_position = [-p for p in camera_pos]
        pred_scene_AP = p.screenshot(return_img=True)
        _ = p.remove_actor(pred)
        # _ = p.remove_actor(pred_box)
        pred_image = cv2.hconcat([pred_scene_PA, pred_scene_AP])

        # ground_truth
        gt = p.add_mesh(tresh_data_gt, cmap=self.listed_color_map, scalars="values", clim=[-0.5, num_classes + 0.5],
                        scalar_bar_args=sargs, smooth_shading=False)
        p.camera_position = camera_pos
        gt_scenePA = p.screenshot(return_img=True)
        p.camera_position = [-p for p in camera_pos]
        gt_sceneAP = p.screenshot(return_img=True)
        _ = p.remove_actor(gt)
        gt_image = cv2.hconcat([gt_scenePA, gt_sceneAP])

        out_image = cv2.vconcat([pred_image, gt_image])

        return out_image

    def log_3dscene_comp_flow(self, volume: np.array, volume_gt: np.array, volume_flow: np.array, num_classes: int = -1,
                              scene_size: int = 480,
                              camera_pos: list = [0.5, -1, 0.5], flow_cmap="RdYlGn", min_max=[0, 1]) -> np.array:

        scene_size = (scene_size,) * 2
        labels = dict(xlabel='R', ylabel='P', zlabel='S')

        if num_classes == -1:
            num_classes = self.classes_num - 1

        data = pv.UniformGrid()
        data_gt = pv.UniformGrid()

        # prediction
        data.dimensions = np.array(volume.shape) + 1
        data.cell_data['values'] = volume.ravel(order='F')
        tresh_data = data.threshold(1, scalars='values')

        # ground_truth
        data_gt.dimensions = np.array(volume_gt.shape) + 1
        data_gt.cell_data['values'] = volume_gt.ravel(order='F')
        tresh_data_gt = data_gt.threshold(1, scalars='values')

        # plotter
        p = pv.Plotter(window_size=scene_size, off_screen=True, lighting='three lights')
        p.set_background('#c1c3e8', top='#7579be')
        p.add_axes(line_width=6, ambient=0.5, **labels)

        sargs = dict(
            title='pulmonary artery',
            title_font_size=16,
            label_font_size=12,
            shadow=True,
            n_labels=self.classes_num,
            italic=False,
            fmt="%.0f",
            font_family="arial",
        )

        # PLOT SCENES
        # prediction
        pred = p.add_mesh(tresh_data, cmap=self.listed_color_map, scalars="values", clim=[-0.5, num_classes + 0.5],
                          scalar_bar_args=sargs, smooth_shading=False)

        p.camera_position = camera_pos
        pred_scene_PA = p.screenshot(return_img=True)
        p.camera_position = [-p for p in camera_pos]
        pred_scene_AP = p.screenshot(return_img=True)
        _ = p.remove_actor(pred)
        pred_image = cv2.hconcat([pred_scene_PA, pred_scene_AP])

        # ground_truth
        gt = p.add_mesh(tresh_data_gt, cmap=self.listed_color_map, scalars="values", clim=[-0.5, num_classes + 0.5],
                        scalar_bar_args=sargs, smooth_shading=False, opacity=0.15)
        flow = p.add_volume(volume_flow, cmap=flow_cmap, clim=min_max)
        p.camera_position = camera_pos
        gt_scenePA = p.screenshot(return_img=True)
        p.camera_position = [-p for p in camera_pos]
        gt_sceneAP = p.screenshot(return_img=True)
        _ = p.remove_actor(gt)
        _ = p.remove_actor(flow)
        gt_image = cv2.hconcat([gt_scenePA, gt_sceneAP])

        out_image = cv2.vconcat([pred_image, gt_image])

        return out_image

    def log_flow(self, flow_img, label):
        ma = abs(np.max(flow_img))
        mi = abs(np.min(flow_img))
        if ma < mi:
            img = flow_img * -1
        else:
            img = flow_img
        ma = np.max(img)
        img[label == 0] = np.nan

        coords = np.argwhere(label)
        x_min, y_min, z_min = coords.min(axis=0)
        x_max, y_max, z_max = coords.max(axis=0)
        label = label[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1]
        img = img[x_min:x_max + 1, y_min:y_max + 1, z_min:z_max + 1]
        return logger.log_3dscene_comp_flow(label, label, img, num_classes=1, min_max=[-ma, ma])


if __name__ == "__main__":
    # from data_augmentation import Transforms
    # from monai.data import Dataset, DataLoader
    # from arg_parser import args
    # from natsort import natsorted
    # import glob
    # import os
    # import matplotlib.pyplot as plt
    #
    # logger = Logger(num_classes=1, is_log_3d=True)
    # data_root_dir = "../../../data/nifti_bssfp_refined/"
    # nifti_paths_scans = natsorted(glob.glob(os.path.join(data_root_dir, '**', '*mri.nii.gz'), recursive=True))
    # nifti_paths_labels = natsorted(
    #     glob.glob(os.path.join(data_root_dir, '**', '*label.nii.gz'), recursive=True))
    # nifti_list = [{args.keys[0]: scan, args.keys[1]: label} for (scan, label) in
    #               zip(nifti_paths_scans, nifti_paths_labels)]
    # trans = Transforms(args)
    # train_dataset = Dataset(nifti_list, trans.train_transform)
    #
    # train_loader = DataLoader(train_dataset, batch_size=1)
    # min_data = np.inf
    # min_label = np.inf
    # max_data = -np.inf
    # max_label = -np.inf
    #
    # for data in train_loader:
    #     img = data['image']
    #     label = data['label']
    #     img = img.squeeze().numpy()
    #     label = label.squeeze().numpy()
    #     view3d = logger.log_3dscene_comp(label, label, num_classes=1)
    #     fig, ax = plt.subplots()
    #     ax.axis("off")
    #     ax.imshow(view3d, interpolation=None)
    #     plt.show()
    #
    # print(f"Min img: {min_data}, max img {max_data}")
    # print(f"Min label: {min_label}, max label {max_label}")

    from data_augmentation import Transforms
    from monai.data import Dataset, DataLoader
    from arg_parser import args
    from natsort import natsorted
    import glob
    import os
    import matplotlib.pyplot as plt

    logger = Logger(num_classes=1, is_log_3d=True)
    data_root_dir = "../../../data/nifti_flow/"
    nifti_paths_scans = natsorted(glob.glob(os.path.join(data_root_dir, '**', '*mri.nii.gz'), recursive=True))
    nifti_paths_labels = natsorted(
        glob.glob(os.path.join(data_root_dir, '**', '*label.nii.gz'), recursive=True))
    nifti_list = [{args.keys[0]: scan, args.keys[1]: label} for (scan, label) in
                  zip(nifti_paths_scans, nifti_paths_labels)]
    trans = Transforms(args)
    train_dataset = Dataset(nifti_list, trans.none_transform)

    train_loader = DataLoader(train_dataset, batch_size=1)

    for data in train_loader:
        img = data['image']
        label = data['label']
        img = img.squeeze().numpy()
        label = label.squeeze().numpy()

        view3d = logger.log_flow(img, label)
        fig, ax = plt.subplots()
        ax.axis("off")
        ax.imshow(view3d, interpolation=None)
        plt.show()
