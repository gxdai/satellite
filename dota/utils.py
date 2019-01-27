import os
import cv2
from glob import glob
import json

from dota import dota_dict


def add_height_width_to_gt(root_dir, gt_folder_name):
    """Add height and width to the ground truth file."""
    
    image_list = glob(os.path.join(root_dir, "images", "*.png"))
    
    output_dir = os.path.join(root_dir, gt_folder_name)


    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)

    for img_path in image_list:
        # read image
        image = cv2.imread(img_path)
        height, width, channel = image.shape
        gt_path = img_path.replace("images", "labelTxt").replace(".png", ".txt")
        with open(gt_path, 'r') as fid:
            lines = fid.readlines()
        lines.insert(2, str(height) + " " + str(width)+'\n')
    
        new_path = gt_path.replace("labelTxt", gt_folder_name)
        with open(new_path, "w") as fid:
            fid.write("".join(lines))



def convert_txt_to_json(gt_dir, json_file):
    """
    This is for converting dota dataset ground truth into json_file
    """
    gt_list = glob(os.path.join(gt_dir, "*.txt")) 
    data_info = dict()

    # three parts of json file, same as coco
    data_info['annotations'] = []
    data_info['categories'] = []
    data_info['images'] = []
    
    # counting the instance id
    instance_idx = 0
    for image_id, gt_file in enumerate(gt_list):
        # read the ground truth file
        image_dict = dict()
        
        image_dict['image_id'] = image_id+1
        image_dict['filename'] = gt_file.split('/')[-1].replace('.txt', '.png') 
  
        data_info['images'].append(image_dict)
        with open(gt_file, 'r') as fid:
            lines = fid.readlines()

        # clean information
        image_source = lines[0] 
        gsd = lines[1]
        height, width = list(map(int, lines[2].split(' ')))
        annotations = lines[3:]
        
        for ann in annotations:
            instance_idx += 1
            ann_dict = dict()

            ann = ann.split(' ')
            ann_dict['id'] = instance_idx 
            ann_dict['image_id'] = image_id + 1
            ann_dict['segmentation'] = list(map(int, ann[:8]))
            ann_dict['category_id'] = dota_dict[ann[8]]
            ann_dict['height'] = height
            ann_dict['width'] = width
            
            data_info['annotations'].append(ann_dict)
      
    with open(json_file, 'w') as fid:
        json.dump(data_info, fid)        
            

def convert_txt_to_json_simple(gt_dir, json_file):
    """
    This is for converting dota dataset ground truth into json_file
    """
    gt_list = glob(os.path.join(gt_dir, "*.txt")) 
    data_info = []

    # counting the instance id
    for image_id, gt_file in enumerate(gt_list):
        # read the ground truth file
        print(gt_file)
        image_dict = dict()
        
        image_dict['image_id'] = image_id+1
        image_dict['image_path'] = gt_file.replace('label_update', 'images').replace('.txt', '.png') 
        image_dict['filename'] = gt_file.split('/')[-1].replace('.txt', '.png') 
  
        with open(gt_file, 'r') as fid:
            lines = fid.readlines()

        # clean information
        image_source = lines[0] 
        gsd = lines[1]
        height, width = list(map(int, lines[2].split(' ')))
        image_dict['height'] = height 
        image_dict['width'] = width

        annotations = lines[3:]
        polygon_list = []
        object_list = []
        
        for ann in annotations:
            ann = ann.split(' ')
            polygon_list.append(list(map(int, ann[:8])))
            object_list.append(dota_dict[ann[8]])
        
        image_dict['polygon_list'] = polygon_list
        image_dict['object_list'] = object_list

        data_info.append(image_dict) 

        
    with open(json_file, 'w') as fid:
        json.dump(data_info, fid)        

if __name__ == '__main__':
    
    # update training labels (add height and width)
    """
    gt_folder_name = 'label_update'
    root_dir = "/home/ubuntu/data/dota/train"
    add_height_width_to_gt(root_dir, gt_folder_name)
    gt_folder_name = 'label_update'
    root_dir = "/home/ubuntu/data/dota/val"
    add_height_width_to_gt(root_dir, gt_folder_name)
    """
    gt_root_dir = "/home/ubuntu/data/dota/train/label_update"
    json_file = "groundtruth.json"
    convert_txt_to_json_simple(gt_root_dir, json_file)
