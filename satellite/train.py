import os
import sys
import argparse
import random
import math
import re
import time
import numpy as np
import cv2
import json
import matplotlib
import matplotlib.pyplot as plt
import skimage.transform
import time

parser = argparse.ArgumentParser(description='For satellite image dataset')
parser.add_argument('--checkpoint_dir', type=str, default='logs/model_65_001_007',
                    help='directory for saving all the checkpoint file.')

parser.add_argument('--dataset_name', type=str, default='satellite')
parser.add_argument('--mode', type=str, default="train")
parser.add_argument('--dup_flag', type=int, default=0)
parser.add_argument('--train_json', type=str, default='complete.json')
# parser.add_argument('--train_json', type=str, default='train.json')
parser.add_argument('--test_json', type=str, default='test.json')
parser.add_argument('--image_root_dir', type=str, default='./')
parser.add_argument('--instance_number_threshold', type=int, default=10,
                    help='delete the category with instance number less than #')
def main():
    ROOT_DIR = os.path.abspath("../")

    # Import Mask
    sys.path.append(ROOT_DIR)
    from mrcnn.config import Config
    from mrcnn import utils
    import mrcnn.model as modellib
    from mrcnn import visualize
    from mrcnn.model import log
    from satellite import SatelliteDataset, SatelliteConfig

    args = parser.parse_args()
    checkpoint_dir = args.checkpoint_dir 
    dataset_name = args.dataset_name
    train_json = args.train_json
    test_json = args.test_json
    image_root_dir = args.image_root_dir
    mode = args.mode
    dup_flag = args.dup_flag
    instance_number_threshold = args.instance_number_threshold

    MODEL_DIR = os.path.join(ROOT_DIR, checkpoint_dir)

    COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
    print("COCO_MODEL_PATH = {}".format(COCO_MODEL_PATH))
    # Download coco trained weights from Releases if needed
    if not os.path.exists(COCO_MODEL_PATH):
        utils.download_trained_weights(COCO_MODEL_PATH)

    # config dataset
    config = SatelliteConfig()
    # config.NUM_CLASSES = 31
    # re-initialize the config class
    # config.__init__()
    config.display()


    dataset_train = SatelliteDataset(dup_flag=dup_flag)
    dataset_train.config_dataset_by_super_category(dataset_name=dataset_name, 
                             json_file=train_json,
                             skip_classes=['Road','Body_Of_water','Tree'],
                             root_dir=image_root_dir,
                             instance_number_threshold=instance_number_threshold,
                             area_set=['001', '002', '003',\
                                       '004', '005', '006',\
                                       '007']
                              )
 

    print(len(dataset_train.image_info))
    # prepare dataset
    dataset_train.prepare()

    dataset_val = SatelliteDataset(dup_flag=False)
    dataset_val.config_dataset_by_super_category(dataset_name=dataset_name, 
                             json_file=train_json,
                             skip_classes=['Road','Body_Of_water','Tree'],
                             root_dir=image_root_dir,
                             instance_number_threshold=instance_number_threshold,
                             area_set=['001', '002', '003',\
                                       '004', '005', '006',\
                                       '007']
                             )

    dataset_val.prepare()
    print(len(dataset_val.image_info))
    print(len(dataset_val.image_info))

    assert True, "stop"

    # create model
    model = modellib.MaskRCNN(mode='training', config=config, model_dir=MODEL_DIR)
    init_with = 'coco'

    if init_with == 'imagenet':
        model.load_weights(model.get_imagenet_weights(), by_name=True)
    elif init_with == 'coco':
        model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=['mrcnn_class_logits', 'mrcnn_bbox_fc', 'mrcnn_bbox', "mrcnn_mask"])
    elif init_with == 'last':
        model.load_weights(model.find_last(), by_name=True)

    # Skip training, Only do validation.
    if mode == 'train': 
        model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE,
            epochs=5,
            layers='heads')
        # fine-tune
        model.train(dataset_train, dataset_val,
            learning_rate=config.LEARNING_RATE /10,
            epochs=10,
            layers='all')
        print("Finish training")

    print("Validation")
    class InferenceConfig(SatelliteConfig):
        GPU_COUNT = 1
        IMAGES_PER_GPU = 1
    inference_config = InferenceConfig()
    # inference_config.GPU_COUNT = 1
    # inference_config.IMAGES_PER_GPU = 1
    # inference_config.BATCH_SIZE = 1
    model = modellib.MaskRCNN(mode='inference',
                          config=inference_config,
                          model_dir=MODEL_DIR)

    model_path = model.find_last()
    print("Load weghts from ", model_path)
    model.load_weights(model_path, by_name=True)



# Evaluation

    APs = []
    # for image_id in dataset_val.image_ids:
    for image_id in dataset_train.image_ids:
        begin_time = time.time()
        
        start_time = time.time()
        image, image_meta, gt_class_id, gt_bbox, gt_mask = \
            modellib.load_image_gt(dataset_train, inference_config,
                                   image_id, use_mini_mask=False)
        print("Data loading time: {}".format(time.time() - start_time))
        molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
        # print(model.config.BATCH_SIZE)
        start_time = time.time()
        resuts = model.detect([image], verbose=0)
        print("Inference time: {}".format(time.time() - start_time))
        r = resuts[0]
        start_time = time.time()
        AP, precisions, recalls, overlaps = \
            utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                    r['rois'], r["class_ids"], r["scores"], r["masks"])
        print("Evaluation time: {}".format(time.time() - start_time))
        APs.append(AP)
        print("overall time: {}".format(time.time() - begin_time))

    print("mAP: ", np.mean(APs))

if __name__ == '__main__':
    main()
