import os
from tqdm import tqdm
import sys
from enum import Enum
import copy
import pickle
import numpy as np


sys.path.insert(0, "/home/gxdai/Documents/mask_rcnn")

# ==== FOr 001 ====== #
# instance_number_threshold = 3
# area_set = ['001']

# For all the 001-007
instance_number_threshold = 1
area_set = ['001', '002', '003', '004', '005', '006', '007']

from service.GeoImage import GeoImage
from service.MaskRCNN import MaskRCNN
from service.ContourProcessor import ContourProcessor

class Cache(Enum):
    none = 1
    read = 2
    write = 3


cacheStrategy = Cache.write

modelName = "MRCNN"

"""
modelVersion = 1
modelStorePath = '/home/gxdai/Documents/mask_rcnn/checkpoint/MRCNN/1/mask_rcnn_satellite_0005_20190109.h5'
modelVersion = 2

modelStorePath = '/home/gxdai/Documents/mask_rcnn/checkpoint/MRCNN/2/mask_rcnn_satellite_20190111.h5'
"""

modelVersion = "super_category"

modelStorePath = '/home/gxdai/Documents/mask_rcnn/checkpoint/MRCNN/super_category/mask_rcnn_satellite_0010.h5'

# modelVersion = "version001"
# modelStorePath = "/home/gxdai/Documents/mask_rcnn/checkpoint/MRCNN/version001/mask_rcnn_satellite_0010.h5"



"""
image_folder = '/home/gxdai/ssd/002_007_TIF'
output_folder = '/home/gxdai/ssd/002_007_GEO' 
image_name = '001_client_region.tif'
"""

# image_name = '002_DG_Satellite_AZ_Airfield_20180818.tif'
# image_name = '003_DG_Satellite_DXB_20180612.tif'
# image_name = '004_DG_Satellite_Fallmouth_Boats_20171006.tif'
# image_name = '005_DG_Satellite_Norfolk_East_20170601_A1.tif'
# image_name = '007_DG_Satellite_NM_Airfield_20171121.tif'


image_folder = '/home/gxdai/ssd/Extra_Experiment_TIF'
output_folder = '/home/gxdai/ssd/Extra_Experiment_GEO'
# image_name = 'DG_Satellite_Norfolk_East_20170601_B1.tif'
image_name = 'DG_Satellite_Rotterdam_Port_Central_20180724.tif'


image_path = os.path.join(image_folder, image_name)

outputDir = os.path.join(output_folder, image_name.split('.')[0])
outputDir = os.path.join(outputDir,'{}/{}/'.format(modelName, modelVersion))

os.makedirs(outputDir, exist_ok=True)
cache_path = os.path.join(outputDir,'full_results.pkl')

geoImg = GeoImage(image_path, gap=200)
maskModel = None if cacheStrategy == Cache.read else MaskRCNN(modelStorePath,
                                                              instance_number_threshold,
                                                              area_set)

geojson = ContourProcessor(geoImg, outputDir)

readCacheIterator = None
if cacheStrategy == Cache.read:
    with open(cache_path, 'rb') as input:
        cachedReadResults = pickle.load(input)
    readCacheIterator = iter(cachedReadResults)


cacheWriteResults = []
for xyOffset in tqdm(geoImg.getSplits()):

    left, up = xyOffset

    if cacheStrategy == Cache.read:
        result = next(readCacheIterator)
    else:
        img = geoImg.getCv2ImgFromSplit(xyOffset)
        print(img.shape)
        print(np.max(img))
        result = maskModel.infere(img, imageId='left-{}_up-{}'.format(left, up))
        if cacheStrategy == Cache.write: cacheWriteResults.append(copy.deepcopy(result))

    geojson.addPatchBoundary(left, up)
    for r in result:
        geojson.addFeature(left, up, r)
        # print("Feature added")

geojson.cleanUp()

if cacheStrategy == Cache.write:
    with open(cache_path, 'wb') as output:
        pickle.dump(cacheWriteResults, output, pickle.HIGHEST_PROTOCOL)
