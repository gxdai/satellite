import os
import sys
from random import shuffle
import json
import numpy as np
import cv2
from glob import glob
import skimage

ROOT_DIR = os.path.abspath("../")
# Import Mask
sys.path.append(ROOT_DIR)

from mrcnn import utils
from mrcnn.config import Config



dota_dict = {
             "plane": 1, 
             "ship": 2,
             "storage-tank": 3, 
             "baseball-diamond": 4,
             "tennis-court": 5,
             "basketball-court": 6,
             "ground-track-field": 7,
             "harbor": 8,
             "bridge": 9,
             "large-vehicle": 10,
             "small-vehicle": 11,
             "helicopter": 12,
             "roundabout": 13, 
             "soccer-ball-field": 14,
             "swimming-pool": 15
            }



class DotaDataset(utils.Dataset): 
    def __init__(self, dup_flag=False): 
        """ Prepare dataset. """
        # This is just a template version.
        super(DotaDataset, self).__init__()
        self.dup_flag = dup_flag
        

    def config_dataset(self, dataset_name=None, json_file=None):
        """
        Args:
            dataset_name: string
            json_file: string
            skip_classes: List[string]
        """
       

        self.class_dict = dota_dict
        

        with open(json_file) as fid:
            data_info = json.load(fid)
        for idx, image_info in enumerate(data_info):
            self.add_image(source=dataset_name, image_id=idx,
                       path=image_info['image_path'],
                       object_list=image_info['object_list'],
                       polygon_list=image_info['polygon_list'],
                       height=image_info['height'],
                       width=image_info['width']
                       )

        # add all the class info
        for category in self.class_dict:
            self.add_class(dataset_name, 
                           self.class_dict[category], 
                           category)
        


    def load_image(self, image_id):
        image = super(DotaDataset, self).load_image(image_id) 
        # resize the image for preprocessing
        # print(image.shape)
        image = cv2.resize(image, (512, 512))
        # print(image.shape) 
        return image

    def load_mask(self, image_id):
        """
        ============================================================
           
        ===========================================================
        """
        
        height, width = self.image_info[image_id]['height'], \
                        self.image_info[image_id]['width']
        # based image path, and get the groundtruth path
        object_list = self.image_info[image_id]['object_list']
        polygon_list = self.image_info[image_id]['polygon_list']
        
        mask_array = []
        for label, polygon in zip(object_list, polygon_list):
            black_ground = np.zeros((height, width))
            # ======================================== # 
            cv2.fillPoly(black_ground, np.array([np.array(polygon).reshape(-1, 2)]).astype(int), (255.,))
            black_ground = cv2.resize(black_ground, (512, 512))
            mask_array.append(black_ground)
        
        mask_array = np.array(mask_array).transpose(1, 2, 0).astype(bool)
        class_id_array = np.array(object_list)
        # print(mask_array.shape)

        return mask_array, class_id_array
        



# Configuration
class DotaConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overwrides values specific to the toy shapes dataset.
    """
    # Give the configuration a recognizable name.
    NAME = "dota"

    # Train on 1 GPU and 8 images per GPU. We put multiple images on each GPU because the images are small.
    # BATCH_SIZE = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

 
    NUM_CLASSES = 1 + 15  # background + # classes

    # Use small images for fast traning. Set the limits of the small side and the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 512
    IMAGE_MAX_DIM = 640
    # Use smaller anchors because our images and objects are small.
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)
    # RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Reduce ROIs per image because the images are small and have few objects. Aim to allow ROI sampling to pick 33%  positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple.
    STEPS_PER_EPOCH = 300
    # Use a small validation steps since the epoch is small
    VALIDATION_STEPS = 5

class InferenceConfig(DotaConfig): 
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


if __name__ == '__main__':
    
    dataset_train = DotaDataset(root_dir="/home/gxdai/ssd/dota/train/images")
    dataset_train.config_dataset(dataset_name='satellite', gt_dir="label_update") 
    dataset_train.prepare()
    
    image_id = 1
    print(dataset_train.image_info[image_id])
    img  = dataset_train.load_image(image_id)
    mask  = dataset_train.load_mask(image_id)


