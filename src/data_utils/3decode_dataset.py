import os
import yaml
import random
import numpy as np
import nibabel as nib
from tqdm import tqdm
from argparse import Namespace
from raster_geometry import sphere, ellipsoid, cuboid, cylinder
from typing import Optional, Sequence, Tuple, Union
import radiomics
import itertools
import SimpleITK as sitk

class ShapeFeaturesCalculator():
    def __init__(self, config_file_path : str):
        # EXCTRACTOR SETUP
        radiomics.setVerbosity(60)
        self.extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(config_file_path)
        feature_names = self.extractor.enabledFeatures['shape']
        print(f"Enabled image types: {self.extractor.enabledImagetypes}")
        print(f"Enabled features: {feature_names}")

    def get_features(self, array: np.array):
        array_itk = sitk.GetImageFromArray(array)
        result = self.extractor.execute(array_itk, array_itk)
        features = []
        for key, v in result.items():
            if 'original_shape' in key:
                if isinstance(v, np.ndarray):
                    v = float(v)
                features.append(v)
        return features
                

class ShapesGenerator():
    def __init__(self, s : float = 1.0, base : int = 5):
        size_ratios = [1.0, 1.5, 2.0]
        shapes, radii  = tuple(int(ratio*s*base*2) for ratio in [1.0, 1.5, 2.0]), tuple(int(ratio*s*base) for ratio in size_ratios)
        
        self.sphere_S = np.array(sphere(shapes[0], radii[0])).astype(int)
        self.sphere_M = np.array(sphere(shapes[1], radii[1])).astype(int)
        self.sphere_L = np.array(sphere(shapes[2], radii[2])).astype(int)
        
        side = tuple(int(s*2*base*ratio) for ratio in size_ratios)
        self.cube_S = np.array(cuboid(shape=(side[0],)*3, semisides=(side[0],)*3)).astype(int)
        self.cube_M = np.array(cuboid(shape=(side[1],)*3, semisides=(side[1],)*3)).astype(int)
        self.cube_L = np.array(cuboid(shape=(side[2],)*3, semisides=(side[2],)*3)).astype(int)
        
        self.cylinder_S = np.array(cylinder(side[0], side[0], int(side[0]/2), 0)).astype(int)
        self.cylinder_M = np.array(cylinder(side[1], side[1], int(side[1]/2), 0)).astype(int)
        self.cylinder_L = np.array(cylinder(side[2], side[2], int(side[2]/2), 0)).astype(int)
        
class VarShapesGenerator():
    def __init__(self, s : float = 1.0, base : int = 5):
        self.scale = s
        self.base = base
        self.size_ratios = [1.0, 1.5, 2.0]
        self.size_ratios_dict = {'S': 1.0,'M': 1.5, 'L': 2.0}

    def get_sphere(self, size: str = 'S', diff_ratio: int = 0.20):
        base_ratio = self.size_ratios_dict[size]
        vratio = random.uniform(base_ratio-diff_ratio, base_ratio+diff_ratio)
        size = int(vratio*self.scale*self.base*2)
        radius = int(vratio*self.scale*self.base)
        return np.array(sphere(size, radius)).astype(int)
    
    def get_elipsoid(self, size: str = 'S', diff_ratio: int = 0.20):
        base_ratio = self.size_ratios_dict[size]
        ratios = [random.uniform(base_ratio-diff_ratio, base_ratio+diff_ratio) for _ in range(3)]
        ax = int(ratios[0]*self.scale*self.base*2)
        ay = int(ratios[1]*self.scale*self.base*2)
        az = int(ratios[2]*self.scale*self.base*2)
        return np.array(ellipsoid(shape=(ax,ay,az), semiaxes=(ax//2,ay//2,az//2))).astype(int)
    
    def get_cuboid(self, size: str = 'S', diff_ratio: int = 0.20):
        base_ratio = self.size_ratios_dict[size]
        ratios = [random.uniform(base_ratio-diff_ratio, base_ratio+diff_ratio) for _ in range(3)]
        ax = int(ratios[0]*self.scale*self.base*2)
        ay = int(ratios[1]*self.scale*self.base*2)
        az = int(ratios[2]*self.scale*self.base*2)
        return np.array(cuboid(shape=(ax,ay,az), semisides=(ax//2,ay//2,az//2))).astype(int)
    
    def get_cylinder(self, size: str = 'S', diff_ratio: int = 0.20):
        base_ratio = self.size_ratios_dict[size]
        vratio = random.uniform(base_ratio-diff_ratio, base_ratio+diff_ratio)
        size = int(vratio*self.scale*self.base*2)
        radius = int(vratio*self.scale*self.base)
        return np.array(cylinder(size, size, radius, 0)).astype(int)
                        
        
class ExperimentGenerator():
    def __init__(self, config):
        config.save_dir = os.path.join(config.save_dir, config.experiment_name)
        config.conditions_dir = os.path.join(config.conditions_dir, config.experiment_name)
        if not os.path.isdir(config.conditions_dir):
            os.makedirs(config.conditions_dir)
        self.config = config
        self.sGen = ShapesGenerator(config.object_scale, config.base_size)
        self.visual_word = np.zeros(
            shape=config.visual_word_length, dtype=np.int32)
        if config.calculate_features:
            self.sCond = ShapeFeaturesCalculator(config.pyradiomics_config_file)

        print(config)
        print("Generating dataset...")
        if config.experiment_name == "size":
            random.seed(config.seed)
            np.random.seed(config.seed)
            images, labels, conditions, shape_features = self.generate_data_size()
        elif config.experiment_name == "shape":
            random.seed(config.seed)
            np.random.seed(config.seed)
            images, labels, conditions, shape_features = self.generate_data_shape()
        elif config.experiment_name == "sizesOfShapes":
            random.seed(config.seed)
            np.random.seed(config.seed)
            images, labels, conditions, shape_features = self.generate_data_sizeOfShapes()  
        elif config.experiment_name == "varSize":
            random.seed(config.seed)
            np.random.seed(config.seed)
            images, labels, conditions, shape_features = self.generate_data_varying_size()  
        elif config.experiment_name == "varSizeAndShape":
            random.seed(config.seed)
            np.random.seed(config.seed)
            images, labels, conditions, shape_features = self.generate_data_varying_shapeAndSize()  
            
        print("Saving dataset...")
        if self.config.file_format == 'nifti':
            self.save_as_nifti(images, labels, pixdim=1.0)
        self.save_data_yaml(conditions)
        
        print("Saving conditioning parameters...")
        if config.calculate_features:
            features_array = np.array(shape_features, dtype=np.float32)
            np.save(os.path.join(config.conditions_dir, 'shape_features.npy'), features_array)
            max_feature = np.amax(features_array, axis=0)
            min_feature = np.amin(features_array, axis=0)
            nominator = (max_feature - min_feature)
            #avoid zero division for minimal features 0 value
            nominator[nominator==0]=1.0 
            features_array_norm = (features_array - min_feature) / nominator
            np.save(os.path.join(config.conditions_dir, 'shape_features_norm.npy'), features_array_norm)
        np.save(os.path.join(config.conditions_dir, "conditions.npy"), np.array(conditions))
        
        print("Dataset generated and saved.")
        
    #SIZE
    def generate_data_size(self) -> Sequence[np.array]:
        shape_condition_vectors = None
        #SOLIDS
        solid_s = self.sGen.sphere_S    
        solid_m = self.sGen.sphere_M
        solid_l = self.sGen.sphere_L
        if self.config.calculate_features: 
            shape_condition_vectors = []
            shape_features_s = self.sCond.get_features(solid_s)
            shape_features_m = self.sCond.get_features(solid_m)
            shape_features_l = self.sCond.get_features(solid_l)
        condition_vectors = []
        images = []
        labels = []
        for _ in tqdm(range(self.config.dataset_size)):
            canvas_image = np.zeros(
                shape=self.config.canvas_size, dtype=np.int32)
            
            # CONDITIONS
            conditions = [[1,0,0,0,0,0],
                          [0,1,0,0,0,0],
                          [0,0,1,0,0,0]]
            conditions_int = [0,1,2]        
            
            #INPUTS
            # small_spheres
            positions_s = []
            for i in range(self.config.objects_number+1):
                pos = self.get_rand_pos()
                canvas_image = self.paintSolid(solid_s, canvas_image, pos)
                positions_s.append(pos)
            
            # medium_spheres
            positions_m = []
            for i in range(self.config.objects_number):
                pos = self.get_rand_pos()
                canvas_image = self.paintSolid(solid_m, canvas_image, pos)
                positions_m.append(pos)
            
            # large_spheres
            positions_l = []
            for i in range(self.config.objects_number-1):
                pos = self.get_rand_pos()
                canvas_image = self.paintSolid(solid_l, canvas_image, pos)
                positions_l.append(pos)
            
            #LABELS
            shape_features = []
            canvas_labels = []
            for cond_int in conditions_int:
                #fresh canvas for labels
                canvas_label = np.zeros(shape=self.config.canvas_size, dtype=np.int32)
                # small_spheres
                if cond_int == 0:
                    for pos in positions_s:
                        canvas_label = self.paintSolid(solid_s, canvas_label, pos)
                    if self.config.calculate_features: 
                        shape_features.append(shape_features_s)
                # medium_spheres
                if cond_int == 1:
                    for pos in positions_m:
                        canvas_label = self.paintSolid(solid_m, canvas_label, pos)
                    if self.config.calculate_features: 
                        shape_features.append(shape_features_m)
                # large_spheres
                if cond_int == 2:
                    for pos in positions_l:
                        canvas_label = self.paintSolid(solid_l, canvas_label, pos)
                    if self.config.calculate_features: 
                        shape_features.append(shape_features_l)
                canvas_labels.append(canvas_label)

            if self.config.calculate_features: 
                shape_condition_vectors.extend(shape_features)
            condition_vectors.extend(conditions)
            images.extend([canvas_image for _ in range(len(conditions))])
            labels.extend(canvas_labels)
                   
        return images, labels, condition_vectors, shape_condition_vectors
    
    #VARYING SIZE
    def generate_data_varying_size(self) -> Sequence[np.array]:
        self.vsGen = VarShapesGenerator(config.object_scale, config.base_size)
        shape_condition_vectors = None
        if self.config.calculate_features: 
            shape_condition_vectors = []
        condition_vectors = []
        images = []
        labels = []
        for _ in tqdm(range(self.config.dataset_size)):
            canvas_image = np.zeros(shape=self.config.canvas_size, dtype=np.int32)
            # CONDITIONS
            conditions = [[1,0,0,0,0,0],
                          [0,1,0,0,0,0],
                          [0,0,1,0,0,0]]
            conditions_int = [0,1,2]        
            
            #INPUTS
            # small_spheres
            positions_s = []
            solids_s = []
            for i in range(self.config.objects_number):
                pos = self.get_rand_pos()
                solid = self.vsGen.get_sphere('S')
                canvas_image = self.paintSolid(solid, canvas_image, pos)
                positions_s.append(pos)
                solids_s.append(solid)
            
            # medium_spheres
            positions_m = []
            solids_m = []
            for i in range(self.config.objects_number):
                pos = self.get_rand_pos()
                solid = self.vsGen.get_sphere('M')
                canvas_image = self.paintSolid(solid, canvas_image, pos)
                positions_m.append(pos)
                solids_m.append(solid)
            
            # large_spheres
            positions_l = []
            solids_l = []
            for i in range(self.config.objects_number):
                pos = self.get_rand_pos()
                solid = self.vsGen.get_sphere('L')
                canvas_image = self.paintSolid(solid, canvas_image, pos)
                positions_l.append(pos)
                solids_l.append(solid)
            
            #LABELS
            shape_features = []
            onehot_vectors = []
            canvas_labels = []
            for cond_int in conditions_int:
                #fresh canvas for labels
                canvas_label = np.zeros(shape=self.config.canvas_size, dtype=np.int32)
                # small_spheres
                if cond_int == 0:
                    features=[]
                    for (pos, solid) in zip(positions_s, solids_s):
                        canvas_label = self.paintSolid(solid, canvas_label, pos)
                        if self.config.calculate_features: 
                            features.append(self.sCond.get_features(solid))
                    onehot_vectors.append(list(itertools.chain(*[conditions[0] for i in range(len(solids_s))])))
                    shape_features.append(list(itertools.chain(*features)))
                # medium_spheres
                elif cond_int == 1:
                    features=[]
                    for (pos, solid) in zip(positions_m, solids_m):
                        canvas_label = self.paintSolid(solid, canvas_label, pos)
                        if self.config.calculate_features: 
                            features.append(self.sCond.get_features(solid))
                    onehot_vectors.append(list(itertools.chain(*[conditions[1] for i in range(len(solids_m))])))
                    shape_features.append(list(itertools.chain(*features)))
                # large_spheres
                elif cond_int == 2:
                    features=[]
                    for (pos, solid) in zip(positions_l, solids_l):
                        canvas_label = self.paintSolid(solid, canvas_label, pos)
                        if self.config.calculate_features: 
                            features.append(self.sCond.get_features(solid))
                    onehot_vectors.append(list(itertools.chain(*[conditions[2] for i in range(len(solids_l))])))
                    shape_features.append(list(itertools.chain(*features)))
                canvas_labels.append(canvas_label)

            if self.config.calculate_features: 
                shape_condition_vectors.extend(shape_features)
            condition_vectors.extend(onehot_vectors)
            images.extend([canvas_image for _ in range(len(canvas_labels))])
            labels.extend(canvas_labels)
                   
        return images, labels, condition_vectors, shape_condition_vectors
    
    #SHAPE
    def generate_data_shape(self) -> Sequence[np.array]:
        shape_condition_vectors = None
        #SOLIDS
        sphere = self.sGen.sphere_M    
        cube = self.sGen.cube_M
        cylinder = self.sGen.cylinder_M
        if self.config.calculate_features: 
            shape_condition_vectors = []
            shape_features_cond1 = self.sCond.get_features(sphere)
            shape_features_cond2 = self.sCond.get_features(cube)
            shape_features_cond3 = self.sCond.get_features(cylinder)
        condition_vectors = []
        images = []
        labels = []
        for _ in tqdm(range(self.config.dataset_size)):
            canvas_image = np.zeros(
                shape=self.config.canvas_size, dtype=np.int32)
            
            # CONDITIONS
            conditions = [[0,0,0,1,0,0],
                          [0,0,0,0,1,0],
                          [0,0,0,0,0,1]]
            conditions_int = [3,4,5]        
            
            #INPUTS
            # spheres
            positions_cond1 = []
            for i in range(self.config.objects_number+1):
                pos = self.get_rand_pos()
                canvas_image = self.paintSolid(sphere, canvas_image, pos)
                positions_cond1.append(pos)
            
            # cubes
            positions_cond2 = []
            for i in range(self.config.objects_number):
                pos = self.get_rand_pos()
                canvas_image = self.paintSolid(cube, canvas_image, pos)
                positions_cond2.append(pos)
            
            # cylinders
            positions_cond3 = []
            for i in range(self.config.objects_number-1):
                pos = self.get_rand_pos()
                canvas_image = self.paintSolid(cylinder, canvas_image, pos)
                positions_cond3.append(pos)
            
            #LABELS
            shape_features = []
            canvas_labels = []
            for cond_int in conditions_int:
                #fresh canvas for labels
                canvas_label = np.zeros(shape=self.config.canvas_size, dtype=np.int32)
                # spheres
                if cond_int == 3:
                    for pos in positions_cond1:
                        canvas_label = self.paintSolid(sphere, canvas_label, pos)
                    if self.config.calculate_features: 
                        shape_features.append(shape_features_cond1)
                # cubes
                if cond_int == 4:
                    for pos in positions_cond2:
                        canvas_label = self.paintSolid(cube, canvas_label, pos)
                    if self.config.calculate_features: 
                        shape_features.append(shape_features_cond2)
                #cylinders
                if cond_int == 5:
                    for pos in positions_cond3:
                        canvas_label = self.paintSolid(cylinder, canvas_label, pos)
                    if self.config.calculate_features: 
                        shape_features.append(shape_features_cond3)
                canvas_labels.append(canvas_label)

            if self.config.calculate_features: 
                shape_condition_vectors.extend(shape_features)
            condition_vectors.extend(conditions)
            images.extend([canvas_image for _ in range(len(conditions))])
            labels.extend(canvas_labels)
                   
        return images, labels, condition_vectors, shape_condition_vectors
    
    #VARYING SHAPE AND SIZE
    def generate_data_varying_shapeAndSize(self) -> Sequence[np.array]:
        vsGen = VarShapesGenerator(config.object_scale, config.base_size)
        shape_condition_vectors = None
        if self.config.calculate_features: 
            shape_condition_vectors = []
        condition_vectors = []
        images = []
        labels = []

        # CONDITIONS
        conditions = [[1, 0, 0, 1, 0, 0],  # 1. small sphere
                      [1, 0, 0, 0, 1, 0],  # 2. small cube
                      [1, 0, 1, 0, 0, 1],  # 3. small cylinder etc.
                      [0, 1, 0, 1, 0, 0],
                      [0, 1, 0, 0, 1, 0],
                      [0, 1, 1, 0, 0, 1],
                      [0, 0, 1, 1, 0, 0],
                      [0, 0, 1, 0, 1, 0],
                      [0, 0, 1, 0, 0, 1],
                      ]
            
        for _ in tqdm(range(self.config.dataset_size)):
            #INPUT
            canvas_image = np.zeros(shape=self.config.canvas_size, dtype=np.int32)
            positions = [self.get_rand_pos() for _ in range(self.config.objects_number * len(conditions))]
            solids = []
            for _ in range(self.config.objects_number):
                for size in ['S', 'M', 'L']:
                    solids.append(vsGen.get_elipsoid(size))
                    solids.append(vsGen.get_cuboid(size))
                    solids.append(vsGen.get_cylinder(size))
            for pos, solid in zip(positions, solids):
                canvas_image = self.paintSolid(solid, canvas_image, pos)
            
            # LABELS
            shape_features = []
            onehot_vectors = []
            canvas_labels = []
            for i in range(3):
                for j in range(3):
                    # fresh canvas for labels
                    canvas_label = np.zeros(shape=self.config.canvas_size, dtype=np.int32)
                    features = []
                    cond_id = i * 3 + j
                    for k in range(self.config.objects_number):
                        pos_id = k * 9 + cond_id
                        solid = solids[pos_id]
                        pos = positions[pos_id]
                        canvas_label = self.paintSolid(
                            solid, canvas_label, pos)
                        if self.config.calculate_features:
                            features.append(self.sCond.get_features(solid))
                    onehot_vectors.append(list(itertools.chain(*[conditions[cond_id] for i in range(self.config.objects_number)])))
                    shape_features.append(list(itertools.chain(*features)))
                    canvas_labels.append(canvas_label)
                    
            if self.config.calculate_features: 
                shape_condition_vectors.extend(shape_features)
            condition_vectors.extend(onehot_vectors)
            images.extend([canvas_image for _ in range(len(canvas_labels))])
            labels.extend(canvas_labels)
        return images, labels, condition_vectors, shape_condition_vectors
    
    #SIZE OF SHAPE
    def generate_data_sizeOfShapes(self) -> Sequence[np.array]:
        shape_condition_vectors = None
        if self.config.calculate_features: 
            shape_condition_vectors = []
        condition_vectors = []
        images = []
        labels = []
        canvas_image = np.zeros(shape=self.config.canvas_size, dtype=np.int32)
        # CONDITIONS
        conditions = [[1, 0, 0, 1, 0, 0],  # 1. small sphere
                      [1, 0, 0, 0, 1, 0],  # 2. small cube
                      [1, 0, 1, 0, 0, 1],  # 3. small cylinder etc.
                      [0, 1, 0, 1, 0, 0],
                      [0, 1, 0, 0, 1, 0],
                      [0, 1, 1, 0, 0, 1],
                      [0, 0, 1, 1, 0, 0],
                      [0, 0, 1, 0, 1, 0],
                      [0, 0, 1, 0, 0, 1],
                      ]
        solids = [self.sGen.sphere_S, self.sGen.cube_S, self.sGen.cylinder_S,
                    self.sGen.sphere_M, self.sGen.cube_M, self.sGen.cylinder_M,
                    self.sGen.sphere_L, self.sGen.cube_L, self.sGen.cylinder_L] 
        if self.config.calculate_features: 
            features = [self.sCond.get_features(solid) for solid in solids]
            
        for _ in tqdm(range(self.config.dataset_size)):
            #INPUT
            canvas_image = np.zeros(shape=self.config.canvas_size, dtype=np.int32)
            positions = [self.get_rand_pos() for _ in range(self.config.objects_number * len(conditions))]
            for pos, solid in zip(positions, solids+solids):
                canvas_image = self.paintSolid(solid, canvas_image, pos)
            
            # LABELS
            shape_features = []
            canvas_labels = []
            for i in range(3):
                for j in range(3):
                    # fresh canvas for labels
                    canvas_label = np.zeros(shape=self.config.canvas_size, dtype=np.int32)
                    cond_id = i * 3 + j
                    for k in range(self.config.objects_number):
                        pos_id = k * 9 + cond_id
                        solid = solids[cond_id]
                        pos = positions[pos_id]
                        canvas_label = self.paintSolid(
                            solid, canvas_label, pos)
                    if self.config.calculate_features:
                        shape_features.append(features[cond_id])
                    canvas_labels.append(canvas_label)
            if self.config.calculate_features: 
                shape_condition_vectors.extend(shape_features)
            condition_vectors.extend(conditions)
            images.extend([canvas_image for _ in range(len(conditions))])
            labels.extend(canvas_labels)
            
        return images, labels, condition_vectors, shape_condition_vectors

    def save_nifti(self, array, path, filename, pixdim=1.0):
        affine_pixdim = np.eye(4) * pixdim
        affine_pixdim[3][3] = 1.0
        nib_array = nib.Nifti1Image(
            array.astype(np.int16), affine=affine_pixdim)
        if not os.path.exists(path):
            os.makedirs(path)
        save_path = os.path.join(path, filename)
        nib.save(nib_array, save_path)

    def save_as_nifti(self, images, labels, pixdim):
        if self.config.experiment_name in ['size', 'shape', 'varSize']:
            n = 3
        else:
            n = 9
        for idx, (img, lbl) in tqdm(enumerate(zip(images, labels)), total=len(images)):
            filename = f'3decode_{idx+1:03d}_{idx % n + 1}.nii.gz'
            self.save_nifti(img, os.path.join(
                self.config.save_dir, 'scans'), filename)
            self.save_nifti(lbl, os.path.join(
                self.config.save_dir, 'labels'), filename)
    
    
    def save_data_yaml(self, conditions):
        #avoid pointer references dump
        yaml.Dumper.ignore_aliases = lambda *args : True
        if self.config.experiment_name in ['size', 'shape']:
            n = 3
        else:
            n = 9
        filenames = [f'3decode_{idx+1:03d}_{idx % n + 1}.nii.gz' for idx in range(self.config.dataset_size * n)]
        data_dict = dict(zip(filenames, conditions))
        with open(os.path.join(self.config.save_dir, 'dataset_list.yaml'), 'w') as outfile:
            yaml.dump(data_dict, outfile, default_flow_style=False, sort_keys=False)

    def save_as_numpy(self, directory, images, labels, conditions, pixdim):
        pass

    def get_rand_pos(self, margin=20):
        x = np.random.randint(margin, self.config.canvas_size[0]-margin)
        y = np.random.randint(margin, self.config.canvas_size[1]-margin)
        z = np.random.randint(margin, self.config.canvas_size[2]-margin)
        return (x, y, z)

    def paintSolid(self, solid, canvas, position):
        x, y, z = position
        w, h, d = solid.shape
        canvas[(x-w//2):(x-w//2+w), (y-h//2):(y-h//2+h), (z-d//2):(z-d//2+d)] = np.where(canvas[(x-w//2):(x-w//2+w), (y-h//2):(y-h//2+h), (z-d//2):(z-d//2+d)] == 0, solid, canvas[(x-w//2):(x-w//2+w), (y-h//2):(y-h//2+h), (z-d//2):(z-d//2+d)])
        return canvas

def generate_all():
    experiments = ['size', 'varSize,' 'shape', 'sizesOfShapes', 'varSizeAndShape']
    for experiment in experiments:
        print (f"Generating dataset for experiment - {experiment}:\n")
        objects_number = 3 
        if experiment == 'sizesOfShapes' or experiment == 'varSizeAndShape':
            objects_number = 2
        config = {"experiment_name": experiment,  # size, shape, sizesOfShapes, 
                "file_format": 'nifti',
                "save_dir": 'data/3decode/nifti_dataset',
                "conditions_dir" : 'data/radiomics/3decode',
                "objects_number": objects_number,
                "canvas_size": [224, 224, 160],
                "object_scale": 1,
                "base_size": 8,
                "visual_word_length": 6,
                "dataset_size": 100,
                'seed': 48,
                'calculate_features': True,
                'pyradiomics_config_file': 'config/radiomics_params_3decode.yaml'}
        config = Namespace(**config)
        eGen = ExperimentGenerator(config)
    

if __name__ == "__main__":
    
    # generate_all() # use to generate all 3decode configurations

    config = {"experiment_name": 'varSize',  # size, varSize, shape, sizesOfShapes, varSizeAndShape 
            "file_format": 'nifti',
            "save_dir": 'data/3decode/nifti_dataset',
            "conditions_dir" : 'data/radiomics/3decode',
            "objects_number": 3, 
            "canvas_size": [224, 224, 160],
            "object_scale": 1,
            "base_size": 8,
            "visual_word_length": 8,
            "dataset_size": 100,
            'seed': 48,
            'calculate_features': True,
            'pyradiomics_config_file': 'config/radiomics_params_3decode.yaml'}
    config = Namespace(**config)
    eGen = ExperimentGenerator(config)