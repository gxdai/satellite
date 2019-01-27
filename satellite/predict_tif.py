from PIL import Image
import os
import sys
import subprocess
import random
import math
import re
import time
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy

ROOT_DIR = os.path.abspath("../")

# Import Mask
sys.path.append(ROOT_DIR)
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

from satellite import SatelliteDataset, SatelliteConfig

# directory to save logs and trained model

# checkpoint_dir = 'logs/model_search'
checkpoint_dir = 'logs/model_20190109'

"""
json_file ="/home/ubuntu/data/satellite/sate/annotations/instances_val2018.json"
model_path = "/home/ubuntu/mask_rcnn/logs/model_search/satellite20190103T0208/mask_rcnn_satellite_0005.h5"
"""



modelName = "MRCNN"
# modelVersion = "super_category"
modelVersion = "big_image"
# modelVersion = "version001"

# parameter settings for local machine

# image_folder = '/home/gxdai/ssd/002_007_TIF'
# output_folder = '/home/gxdai/ssd/002_007_VISUALIZE' 


# image_name = '002_DG_Satellite_AZ_Airfield_20180818.tif'
image_name = '001_client_region.tif'
# image_name = '003_DG_Satellite_DXB_20180612.tif'
# image_name = '004_DG_Satellite_Fallmouth_Boats_20171006.tif'
# image_name = "005_DG_Satellite_Norfolk_East_20170601_A1.tif"
# image_name = "007_DG_Satellite_NM_Airfield_20171121.tif"
# model_path = "/home/gxdai/Documents/mask_rcnn/checkpoint/MRCNN/2/mask_rcnn_satellite_20190111.h5"

# model_path = "/home/gxdai/Documents/mask_rcnn/checkpoint/MRCNN/super_category/mask_rcnn_satellite_0010.h5"
# model_path = "/home/gxdai/Documents/mask_rcnn/checkpoint/MRCNN/version001/mask_rcnn_satellite_0010.h5"

#model_path = "../checkpoint/MRCNN/big_image/mask_rcnn_satellite_0010.h5"
model_path = "../checkpoint/MRCNN/big_image/mask_rcnn_satellite_0005.h5"

"""
image_folder = '/home/gxdai/ssd/Extra_Experiment_TIF'
output_folder = '/home/gxdai/ssd/Extra_Experiment_VISUALIZE' 
"""


image_folder = '/home/ubuntu/data/satellite/002_007_TIF'
output_folder = '/home/ubuntu/data/satellite/002_007_VISUALIZE' 
# image_name = 'DG_Satellite_Norfolk_East_20170601_B1.tif'
# image_name = '001_15JUL05082509-P2AS_R1C1-BAbas20150507-03-N16B-WV01.tif'
# image_name = 'grayJUL05082509-P2AS_R1C1-BAbas20150507-03-N16B-WV01.tif'
# image_name = 'new_001.tif'
# image_name = 'cropped_10.tif'
# image_name = 'cropped_10_render.tif'
# image_name = "DG_Satellite_Norfolk_East_20170601_B1.tif"
# image_name = "DG_Satellite_Rotterdam_Port_Central_20180724.tif"


"""


# parameter settings for DGX
image_folder = '/home/ubuntu/002_007_TIF'
output_folder = '/home/ubuntu/002_007_VISUALIZE' 
image_name = '002_DG_Satellite_AZ_Airfield_20180818.tif'
model_path = "/home/ubuntu/mask_rcnn/checkpoint/MRCNN/2/mask_rcnn_satellite_20190111.h5"
"""



# image_name = '003_DG_Satellite_DXB_20180612.tif'
# image_name = '004_DG_Satellite_Fallmouth_Boats_20171006.tif'
# image_name = '005_DG_Satellite_Norfolk_East_20170601_A1.tif'
# image_name = '007_DG_Satellite_NM_Airfield_20171121.tif'
# image_name = '001_client_region.tif'
image_path = os.path.join(image_folder, image_name)

print("Read the whole image")
# whole_image_tif = plt.imread(image_path)
whole_image_tif = cv2.imread(image_path)
"""
shape = whole_image_tif.shape

hist, bin_edge = np.histogram(whole_image_tif, bins=100)
print(hist / float(shape[0]*shape[1]*shape[2]))
print("*"*10)
print(bin_edge)
sys.exit()
"""
# ======================================= #
# convert .png to .tif
# cv2.imwrite(image_path.replace(".png", ".tif"), whole_image_tif)
# ====================================== #

print(whole_image_tif.shape)
whole_image_tif = cv2.cvtColor(whole_image_tif,cv2.COLOR_BGR2RGB)

print(whole_image_tif.shape)

print("np.max(whole_image_tif) = {}".format(np.max(whole_image_tif)))
print("np.min(whole_image_tif) = {}".format(np.min(whole_image_tif)))
# pred_image_tif = np.zeros((whole_image_tif.shape)).astype(np.uint8)

# whole_image_tif = whole_image_tif * (255 // int(max_pixel_value) + 1)
print("READ iis DONE")
pred_image_tif = copy.deepcopy(whole_image_tif)

# only keep 3 channels
pred_image_tif = pred_image_tif[:, :, :3]




print(isinstance(image_path, str))
print(isinstance(whole_image_tif, np.ndarray))
print(whole_image_tif.shape)

outputDir = os.path.join(output_folder, image_name.split('.')[0])
outputDir = os.path.join(outputDir,'{}/{}/'.format(modelName, modelVersion))

json_file ="../data/complete.json" 



MODEL_DIR = os.path.join(ROOT_DIR, checkpoint_dir)

config = SatelliteConfig()
config.display()

# config = SatelliteConfig()

# prediction for my previous model, which has only 69 classes
# config.NUM_CLASSES=70

# config.__init__()

# =======   For training on 001 only ===================
instance_number_threshold=3 
# =======   otherwise                ==================
# instance_number_threshold=1 

# ================ only select 001 images  ============
area_set = ['001']

# ===============  select all the areas    ===========
# area_set=['001', '002', '003', '004', '005', '006', '007']

""" For old training settings

dataset_val = SatelliteDataset()
dataset_val.config_dataset(dataset_name='satellite', 
                                 json_file=json_file,
                                 skip_classes=['Road','Body_Of_water','Tree'],
                                 root_dir='./001',
                                 instance_number_threshold=instance_number_threshold, 
                                 area_set=area_set 
                            )

"""


dataset_val = SatelliteDataset(mode='evaluation')
dataset_val.config_dataset_with_big_image(dataset_name='satellite', 
                                          root_dir="/home/ubuntu/mask_rcnn/data/big_image"
                                          )






dataset_val.prepare()





class InferenceConfig(SatelliteConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


inference_config = InferenceConfig()
model = modellib.MaskRCNN(mode='inference',
                          config=inference_config,
                          model_dir=MODEL_DIR)
# model_path = model.find_last()
print("Load weghts from ", model_path)
print("Specify model path")
model.load_weights(model_path, by_name=True)



"""
print("Clean up prediction directory")
bash_cmd = "rm -rf " + pred_dir
process = subprocess.Popen(bash_cmd.split(), stdout=subprocess.PIPE)
output, error = process.communicate()
"""

if not os.path.isdir(outputDir):
    os.makedirs(outputDir)

original_image_shape = (1024, 1024)

step = 1024


width_step = 1024
height_step = 1024

height_number = whole_image_tif.shape[0] // height_step - 1
width_number = whole_image_tif.shape[1] // width_step  - 1


print("Slice the whole image, and predict it one by one")

for row in tqdm(range(height_number)):
    for col in range(width_number):
        sliced_img = whole_image_tif[row*height_step:(row+1)*height_step,
                                     col*width_step:(col+1)*width_step]
        original_image = modellib.load_image_only(sliced_img, config)
        # original_image = cv2.resize(original_image, (256, 256))
        results = model.detect([original_image], verbose=1)
        r = results[0]

        pred_image = visualize.save_instances(original_image, r['rois'], r['masks'],
                                          r['class_ids'],dataset_val.class_names, 
                                          r['scores'], 
                                          original_image_shape=original_image_shape) 
        pred_image_tif[row*height_step:(row+1)*height_step,col*width_step:(col+1)*width_step] = pred_image

output_file_path = os.path.join(outputDir, image_name.split('.')[0]+'.tif')

# plt.imsave(output_file_path, pred_image_tif)

# pred_image_tif = cv2.cvtColor(pred_image_tif, cv2.COLOR_RGB2BGR)
pred_image_tif = pred_image_tif[:, :, ::-1] # reverse the channel for cv2.imwrite
cv2.imwrite(output_file_path, pred_image_tif)
