import os
import sys

import copy

from random import shuffle
import json
import numpy as np
import cv2
import pdb
from glob import glob
from tqdm import tqdm

from shapely.geometry import Polygon, mapping
import shapely

ROOT_DIR = os.path.abspath("../")
# Import Mask
sys.path.append(ROOT_DIR)

from mrcnn import utils
from mrcnn.config import Config

import multiprocessing as mp


SKIP_CLASSES = [
                 "Intersection/Crossroads",
                 "Pedestrian",
                 "Airports",
                 "Roundabout",
                 "Shipping Container Lot",
                 "Rail(train)",
                 "Port",
                 "Telephone Poles",
                 "Hovercraft",
                 "Power Station"
               ]



def category_map_name_to_super(json_file):
    """
    Args:
        json_file: The ground truth json_file for patch images.

    Returns:
        category_map: {name: super_name, name: super_name, ...}
    """


    with open(json_file) as fid:
        data_info = json.load(fid)

    category_map = dict()
    for anno in data_info['categories']:
        if anno['name'] not in category_map and anno['supercategory'] not in SKIP_CLASSES:
            category_map[anno['name']] = anno['supercategory']
   
    super_category_set = set([category_map[key] for key in category_map]) 

    return category_map, super_category_set
    




def get_static_info(data_info):
    # Get statics results for each categories
    """
    Missiles:                       1
    Police_Station:                 1
    Rail_(for_train):               8
    Vehicle_Sheds:                  21
    Hospital:                       0
    Storage_Tanks:                  490
    School:                         0
    Satellite_Dish:                 4
    Submarine:                      7
    ...
    ...
    """


    static = dict()
    id_name_map = dict()

    for cls in data_info['categories']:
        if cls['id'] not in id_name_map:
            id_name_map[cls['id']] = cls['name']


    for ins in data_info['annotations']:
        cls_name = id_name_map[ins['category_id']]
        if cls_name not in static:
            static[cls_name] = 0
        else:
            static[cls_name] += 1
    
    return id_name_map, static



# 'Missiles','Satellite_Dish','Submarine', 'Helipads', 'Plane',
def duplicate_images(data_info, duplicate_categories=[
                                     'Tanks',
                                     'Crane',
                                     'Ships'], target_number=500):
    """
    Duplicate the categories with rare instances.
    """
    # get statics results.
    id_name_map, static = get_static_info(data_info)
    
    duplicate_statics = dict()

    for category in duplicate_categories:
        instance_number = static[category]
        dup_times = target_number // instance_number
        if dup_times >= 2:
            duplicate_statics[category] = dup_times
    
    return duplicate_statics


def max_duplicate_counter(duplicate_statics, object_list, category_dict):

    # ********************************************************** #
    #            if there are more than one rare instances in a single image, we duplicate
    #            with the largest number in statics results.
    # ********************************************************** #


    """
    duplicate_statics: {category_name: duplicate times}
    object_list: List[int]

    Returns:
            int
    """
 
    # reverse category dict
    id_category = dict((v, k) for k, v in category_dict.items())
    
    # the number of times we need to duplicate the data 
    counter = 1
    for obj in object_list:
        if id_category[obj] in duplicate_statics:
            counter = max(counter,
                          duplicate_statics[id_category[obj]])

    return counter



def selection_for_specified_area(data_info, area_set=['001']):
    """
    This is for picking specified area.
    """
   
    # Get information for different parts. 
    annotations = data_info['annotations']
    images = data_info['images']
    categories = data_info['categories']
    
    # pick images for specified area
    print("Before selction, len(images)) = {}".format(len(images)))
    images = [image for image in images if image['coco_url'].split('/')[-1][:3] in area_set]
    print("After selction, len(images)) = {}".format(len(images)))
    
    # update annotations parts, selecting instances for the specified areas
    updated_annotations = []
    for anno in annotations:
        # check if this instance exist in this area
        exitence_flag = False
        for image in images:
            if anno['image_id'] == image['id']:
                updated_annotations.append(anno)
                break
    print("Before selection: len(annotations) = {}".format(len(annotations)))
    print("After selection: len(updated_annotations) = {}".format(len(updated_annotations)))
    
    # update data_info
    data_info['annotations'] = updated_annotations
    data_info['images'] = images

    return data_info
    
            
    
    



class SatelliteDataset(utils.Dataset): 
    def __init__(self, 
                 dup_flag=False,
                 mode='train'
                ): 
        """ Prepare dataset. """
        # This is just a template version.
        super(SatelliteDataset, self).__init__()
        self.dup_flag = dup_flag
        self.mode = mode

        
        
    def config_dataset_by_category(self, dataset_name, 
                             json_file, 
                             skip_classes, 
                             root_dir,
                             instance_number_threshold,
                             area_set):
        """
        Args:
            dataset_name: string
            json_file: string
            skip_classes: List[string]
        """
        pdb.set_trace()
        def get_gt_info(idx, annotations, category_dict, updated_category_dict):
            # Get the groundtruth information for each image, including label and polygon
            object_list = []
            polygon_list = []
            
            # Go through all the annotations for certain image_id (idx)
            for anno in annotations:
                if idx == anno['image_id']:
                    category_id = anno['category_id']
                    category_name = category_dict[category_id] 
                    
                    if category_name not in updated_category_dict:
                        # skip all the irrelevant classes.
                        continue
                    object_list.append(updated_category_dict[category_name])
                    poly = anno['segmentation'][0]
                    polygon_list.append([poly[i:i+2] for i in range(0, len(poly), 2)])

            return object_list, polygon_list

        with open(json_file) as f:
            data_info = json.load(f)

        print("class number = {} ".format(len(data_info['categories'])))
        data_info = selection_for_specified_area(data_info, area_set)

        def find_skip_classes(data_info, threshold):
            """
            skip those categories with number of instances less than theshold.
            """
            # create empyt dict 
            static = dict()
            id_name_map = dict()

            for cls in data_info['categories']:
                if cls['id'] not in id_name_map:
                    id_name_map[cls['id']] = cls['name']


            for ins in data_info['annotations']:
                cls_name = id_name_map[ins['category_id']]
                if cls_name not in static:
                    static[cls_name] = 0
                else:
                    static[cls_name] += 1

            skip_category =[]
            for idx in id_name_map:
                category_name = id_name_map[idx]
                if category_name in static:
                    if static[category_name] < threshold:
                        skip_category.append(category_name)
                else:
                    skip_category.append(category_name)

            return skip_category

        # skip the category with 0 instances
        print("skip categories with instance number less than {}".format(instance_number_threshold))
        skip_classes = find_skip_classes(data_info, threshold=instance_number_threshold)            
        print("DONE")
        # skip_classes = ['Running_Track', 'Flag', 'Train', 'Cricket_Pitch',\
        #                 'Horse_Track', 'Artillery', 'Racing_Track', \
        #                 'Power_Station', 'Refinery', 'Mosques', 'Prison',\
        #                'Market/Bazaar', 'Quarry', 'School', 'Graveyard',\
        #                'Rifle_Range', 'Train_Station', 'Telephone_Line',\
        #                'Vehicle_Control_Point', 'Hospital']

        def get_category_dict(categories, 
                              skip_classes=['Road','Body_Of_water','Tree']):
           
            category_dict = dict() 
            updated_category_dict = dict() 
            id_counter = 1
            for category in categories:
                if category['id'] not in category_dict:
                     category_dict[category['id']] = category['name']
                
                if category['name'] not in updated_category_dict \
                                 and category['name'] not in skip_classes:
                     updated_category_dict[category['name']] = id_counter
                     id_counter += 1 

            return category_dict, updated_category_dict


        category_dict, updated_category_dict = get_category_dict(
                                                  data_info['categories'],                                                                  skip_classes=skip_classes)

        images_list = data_info['images']
        for i, img_dict in enumerate(images_list):
            image_path = os.path.join(root_dir, img_dict['coco_url'])
            width = img_dict['width']
            height = img_dict['height']
            object_list, polygon_list = get_gt_info(img_dict['id'], 
                                                    data_info['annotations'],
                                                    category_dict,
                                                    updated_category_dict)
            # skip all those empty frames
            if not object_list:
                continue

            if self.dup_flag:
                duplicate_statics = duplicate_images(data_info)
                counter = max_duplicate_counter(duplicate_statics, 
                                                object_list,
                                                updated_category_dict)

                for _ in range(counter):
                    self.add_image(source="satellite", image_id=i,
                       path=image_path,
                       width=width,
                       height=height,
                       object_list=object_list,
                       polygon_list=polygon_list,
                       )
            else:

                self.add_image(source="satellite", image_id=i,
                       path=image_path,
                       width=width,
                       height=height,
                       object_list=object_list,
                       polygon_list=polygon_list,
                       )


        # shuffle the self.image_info list.
        # this happens in place, and returns None
        shuffle(self.image_info)
 
        # add all the class info
        for category in updated_category_dict:
            # self.add_class(dataset_name, category['id'], category['name'])
            self.add_class(dataset_name, 
                           updated_category_dict[category], 
                           category)

        print(len(updated_category_dict))
        print(updated_category_dict)
        print(skip_classes)
        print(len(self.image_info))
        print('\t'*3 + "#"*20 + '\n'*2)
        print('\t'*3 + "CONFIG Daset" + '\n'*2)
        print(len(self.class_info))
        print('\t'*3 + "#"*20 + '\n'*2)
        
 
    def config_dataset_by_super_category(self, dataset_name, 
                             json_file, 
                             skip_classes, 
                             root_dir,
                             instance_number_threshold,
                             area_set):
        """
        Args:
            dataset_name: string
            json_file: string
            skip_classes: List[string]
        """

        def get_gt_info(idx, annotations, category_dict, updated_category_dict):
            # Get the groundtruth information for each image, including label and polygon
            object_list = []
            polygon_list = []
            
            # Go through all the annotations for certain image_id (idx)
            for anno in annotations:
                if idx == anno['image_id']:
                    category_id = anno['category_id']
                    category_name = category_dict[category_id] 
                    
                    if category_name not in updated_category_dict:
                        # skip all the irrelevant classes.
                        continue
                    object_list.append(updated_category_dict[category_name])
                    poly = anno['segmentation'][0]
                    polygon_list.append([poly[i:i+2] for i in range(0, len(poly), 2)])

            return object_list, polygon_list

        with open(json_file) as f:
            data_info = json.load(f)

        print("class number = {} ".format(len(data_info['categories'])))
        data_info = selection_for_specified_area(data_info, area_set)

        def find_skip_classes(data_info, threshold):
            """
            skip those categories with number of instances less than theshold.
            """
            # create empyt dict 
            static = dict()
            id_name_map = dict()

            for cls in data_info['categories']:
                if cls['id'] not in id_name_map:
                    id_name_map[cls['id']] = cls['name']


            for ins in data_info['annotations']:
                cls_name = id_name_map[ins['category_id']]
                if cls_name not in static:
                    static[cls_name] = 0
                else:
                    static[cls_name] += 1

            skip_category =[]
            for idx in id_name_map:
                category_name = id_name_map[idx]
                if category_name in static:
                    if static[category_name] < threshold:
                        skip_category.append(category_name)
                else:
                    skip_category.append(category_name)

            return skip_category

        # skip the category with 0 instances
        print("skip categories with instance number less than {}".format(instance_number_threshold))
        skip_classes = find_skip_classes(data_info, threshold=instance_number_threshold)            
        print("DONE")
        # skip_classes = ['Running_Track', 'Flag', 'Train', 'Cricket_Pitch',\
        #                 'Horse_Track', 'Artillery', 'Racing_Track', \
        #                 'Power_Station', 'Refinery', 'Mosques', 'Prison',\
        #                'Market/Bazaar', 'Quarry', 'School', 'Graveyard',\
        #                'Rifle_Range', 'Train_Station', 'Telephone_Line',\
        #                'Vehicle_Control_Point', 'Hospital']
        skip_classes = [
                        "Intersection/Crossroads",
                        "Pedestrian",
                        "Airports",
                        "Roundabout",
                        "Shipping Container Lot",
                        "Rail(train)",
                        "Port",
                        "Telephone Poles",
                        "Hovercraft",
                        "Power Station"]

        def get_category_dict(categories, 
                              skip_classes=['Road','Body_Of_water','Tree']):
           
            category_dict = dict() 
            updated_category_dict = dict() 
            id_counter = 1
            for category in categories:
                if category['id'] not in category_dict:
                     category_dict[category['id']] = category['supercategory']
                
                if category['supercategory'] not in updated_category_dict \
                                 and category['supercategory'] not in skip_classes:
                     updated_category_dict[category['supercategory']] = id_counter
                     id_counter += 1 

            return category_dict, updated_category_dict


        category_dict, updated_category_dict = get_category_dict(
                                                                data_info['categories'],
                                                                skip_classes=skip_classes)

        images_list = data_info['images']
        for i, img_dict in enumerate(images_list):
            image_path = os.path.join(root_dir, img_dict['coco_url'])
            width = img_dict['width']
            height = img_dict['height']
            object_list, polygon_list = get_gt_info(img_dict['id'], 
                                                    data_info['annotations'],
                                                    category_dict,
                                                    updated_category_dict
                                                   )
            # skip all those empty frames
            if not object_list:
                continue

            if self.dup_flag:
                duplicate_statics = duplicate_images(data_info)
                counter = max_duplicate_counter(duplicate_statics, 
                                                object_list,
                                                updated_category_dict)

                for _ in range(counter):
                    self.add_image(source="satellite", image_id=i,
                       path=image_path,
                       width=width,
                       height=height,
                       object_list=object_list,
                       polygon_list=polygon_list,
                       )
            else:

                self.add_image(source="satellite", image_id=i,
                       path=image_path,
                       width=width,
                       height=height,
                       object_list=object_list,
                       polygon_list=polygon_list,
                       )


        # shuffle the self.image_info list.
        # this happens in place, and returns None
        shuffle(self.image_info)
 
        # add all the class info
        for category in updated_category_dict:
            # self.add_class(dataset_name, category['id'], category['name'])
            self.add_class(dataset_name, 
                           updated_category_dict[category], 
                           category)

        print(len(updated_category_dict))
        print(updated_category_dict)
        print(skip_classes)
        print(len(self.image_info))
        print('\t'*3 + "#"*20 + '\n'*2)
        print('\t'*3 + "CONFIG Daset" + '\n'*2)
        print(len(self.class_info))
        print('\t'*3 + "#"*20 + '\n'*2)
        


    def visualize(self, image_id):
        tmp_dir = './tmp'
        if not os.path.isdir(tmp_dir):
            os.makedirs(tmp_dir)
        img = self.load_image(image_id)
        mask, class_id = self.load_mask(image_id)
        
        color_set = [np.array([0, 0, 255]),
                     np.array([0, 255, 0]),
                     np.array([255, 0, 0]),
                     np.array([255, 255, 0]),
                     np.array([0, 255, 255]),
                     np.array([255, 0, 255])
                    ]
        total_color_number = len(color_set)
        alpha = 0.6
        print(mask.shape)
        fontFace = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        for idx, single_mask in enumerate(mask.transpose(2, 0, 1)):
            index = np.where(single_mask.astype(float)>0.5)
            color_index = np.random.randint(total_color_number)
            img[index] = img[index] * alpha + (1-alpha) * color_set[color_index]
            
            # attach class name for visualization at top left corner of mask
            try:

                min_row = min(index[0])
                min_col = min(index[1])
            except ValueError:
                continue
            
            cv2.putText(img, self.idx_to_name_map[class_id[idx]], (min_col, min_row), fontFace, fontScale,  color_set[color_index].tolist(), lineType=cv2.LINE_AA)
            
            
            
            
        # convert rgb to bgr
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(tmp_dir, str(image_id)+'.png'), img)
        
#         color_set = [[0, 0, 255],
#                      [0, 255, 0],
#                      [255, 0, 0],
#                      [255, 255, 0],
#                      [0, 255, 255],
#                      [255, 0, 255]
#                     ]
        
#         # directly use polygon to for visluation
#         img = self.load_image(image_id)
#         for polygon in self.image_info[image_id]['polygon_list']:
            
#             color_index = np.random.randint(total_color_number)
#             cv2.fillPoly(img, np.array([polygon]).astype(int), color_set[color_index])

        cv2.imwrite(str(image_id)+'_fill_' + '.png', img)
    def load_image(self, image_id):
        if self.image_info[image_id]['image_type'] == 'patch':
            image = super(SatelliteDataset, self).load_image(image_id) 
        elif self.image_info[image_id]['image_type'] == 'big':
            # only pick small patch at specified location
            
            big_image_id = self.image_info[image_id]['big_image_id'] # '001', '002', ....
            big_image = self.image_dict[big_image_id] # numpy array for big image
            start_h = self.image_info[image_id]['start_h']
            start_w = self.image_info[image_id]['start_w']
            patch_height = self.image_info[image_id]['height']
            patch_width = self.image_info[image_id]['width']

            image = big_image[start_h:start_h+patch_height, start_w:start_w+patch_width]
        

        # resize the image for preprocessing
        # print(image.shape)
        # image = skimage.transform.resize(image, (256, 256))
        # print(image.shape) 
        return image


    def load_mask(self, image_id):
        """
        ============================================================
           For this preliminary task, we only work on three classes
           
                        Tree,
                        Vehicles,
                        Buildings,
        ===========================================================
        """
        
        object_list = self.image_info[image_id]['object_list']
        polygon_list = self.image_info[image_id]['polygon_list']
        width = self.image_info[image_id]['width']
        height = self.image_info[image_id]['height']
        # print(self.image_info[image_id]['path'])

        assert len(object_list) == len(polygon_list), "object number and ploy number doesn't match"
      
        mask_array = []
        class_id_array = []
        
        object_number_threshold = 1e8
        
        
        for idx, (label, polygon) in enumerate(zip(object_list, polygon_list)):
            if idx >= object_number_threshold:
                # Too many objects would be out of memory
                break
            
            black_ground = np.zeros((height, width))
            cv2.fillPoly(black_ground, np.array([polygon]).astype(int), (1.,))
#             cv2.fillConvexPoly(black_ground, np.array([polygon]).astype(int), (1.,))
            black_ground = black_ground.astype(bool)
            mask_array.append(black_ground)
            class_id_array.append(object_list[idx])
        
        # mask_array = np.stack(mask_array, axis=0).transpose(1, 2, 0).astype(bool)
        mask_array = np.stack(mask_array, axis=2)
        class_id_array = np.array(class_id_array)
        
        # print(mask_array.shape)
        return mask_array, class_id_array

    def config_dataset_with_big_image(self, 
                                      dataset_name,
                                      root_dir):
        """
        Args:
            root_dir: string. The directory saving all the big images and annotations.
        """
        
        # read the ground truth category information
        category_map, super_category_set = category_map_name_to_super("../data/complete.json")
        # add all the class info
        name_to_idx_map = dict()
        self.idx_to_name_map = dict()
        for idx, category in enumerate(super_category_set):
            # self.add_class(dataset_name, category['id'], category['name'])
            name_to_idx_map[category] = idx+1
            self.idx_to_name_map[idx+1] = category
            
            self.add_class(dataset_name, 
                           idx+1, 
                           category)

        if self.mode == 'evaluation':
            return
        
        def calculate_intersection_over_poly1(poly1, poly2):
            """
            It is not the iou on usual, the iou is the value of intersection over poly1
            """
            # Get the intersection of two polygons.
            inter_poly = poly1.intersection(poly2)
            # area 
            inter_area = inter_poly.area

            poly1_area = poly1.area
            half_iou = inter_area / poly1_area
            # the area of inter-poly over the poly1_area.
            
            
            return half_iou, inter_poly

        def convert_poly_2_numpy(polygon):
            """
            Args:
                polygon: shapely.geometric object
            
            Returns:
                :params numpy 
            """
            #pdb.set_trace()
            try:
                polygon = mapping(polygon)['coordinates']
            except KeyError:
                polygon = mapping(polygon)['geometries']
                for instance in polygon:
                    if instance['type'] == 'Polygon':
                        polygon = instance['coordinates']
                        break

            polygon = polygon[0]
            polygon = np.stack(polygon, axis=0)
            
            return polygon

        def get_class_id_and_polygon(start_point=None,
                                     patch_size=None,
                                     annotation_dict=None,
                                     category_map=None,
                                     name_to_idx_map=None
                                    ):
            """
            Args:
                start_point: (y, x)
                steps: (step_y, step_x)
                annotation_set: polygon and labels
                category_map: map_dict {name: super_category} 
                skip_classes: Llist[supercategory_name]  
           
            Returns:
                object_list and polygon_list
            """


            def convert_polygon_to_box(polygon, box_format='yxyx'):
                """
                Args:
                    polygon: [[x, y], [x, y], [x, y], ...]
                
                Returns
                    box: [y1, x1, y2, x2]
                """
                xy_min = np.amin(polygon, axis=0)
                xy_max = np.amax(polygon, axis=0)
                if box_format == 'xyxy':
                    return np.concatenate([xy_min, xy_max])
                elif box_format == 'yxyx':
                    return np.concatenate([xy_min[::-1], xy_max[::-1]])


            def truncate_polygon(polygon, patch_box):
                """
                Truncate the polygon, to make it inside the images
                
                Args:
                    polygon: [[x,y], [x,y], ...]
                    patch_box: [y1, x1, y2, x2]
                
                Args:
                    polygon: [[x, y], [x, y], ...] ...   x1 <= x <= x2, y1 <= y <= y2
              
                """
                
                # check the x_min 
                patch_width = patch_box[3] - patch_box[1]
                patch_height = patch_box[2] - patch_box[0]
                polygon[:, 0] = np.minimum(np.maximum(polygon[:, 0] - patch_box[1], 0), patch_width)
                polygon[:, 1] = np.minimum(np.maximum(polygon[:, 1] - patch_box[0], 0), patch_height)

                return polygon


            def computer_intersection_over_box2(box1, box2):
                """Computer intersection area over the second box."""
                # box1: The patch image location, format (y1, x1, y2, x2) 
                # box2: the bbox of polygon, format (x1, y1, x2, y2)
                
                y1 = max(box1[0], box2[0])
                x1 = max(box1[1], box2[1])

                y2 = min(box1[2], box2[2])
                x2 = min(box1[3], box2[3])
                

                intersection_area = max(y2-y1, 0) * max(x2-x1, 0)
                box2_area = max(box2[2]-box2[0], 0) * max(box2[3]-box2[1], 0)
               
                area_theshold = 1e-3 # if the polgon is too small, we will ignore it. 
                if box2_area < area_theshold:
                    iou = 0
                else:
                    iou = intersection_area / box2_area
                
                # convert polygon into box by choosing the x_min, y_min, x_max, y_max
                return iou 
            

            # print("len(annotation_dict['objects'])", len(annotation_dict['objects']))
            def check_if_center_inside_patch(center, patch_box):
                """
                check if the center point is inside the patch
               
                Args:
                    center: [x, y]
                    patch_box: [y_min, x1, y2, x2]

                Returs:
                    flag: True for inside, False for outside
                """
                # to make sure the object is completely inside the 
                slack = 0
                
                
                flag = (center[0] >= (patch_box[1]+slack)) and \
                       (center[0] <= (patch_box[3]-slack)) and \
                       (center[1] >= (patch_box[0]+slack)) and \
                       (center[1] <= (patch_box[2]-slack))

                return flag


            
            threshold = 1e-3
            patch_box = start_point + [x+y for x, y in zip(start_point, patch_size)]
            
            object_list = []
            polygon_list = []
            

            threshold_area = 0.1
            
            for index, obj in enumerate(annotation_dict['objects']):
                label = obj['label']
                if label not in category_map:
                    # we should skip this class
                    continue
                else:
                    #pdb.set_trace()
                    
                    # check if the center of polygon is inside the patch
                    flag = check_if_center_inside_patch(obj['center'], patch_box)
                    if not flag:
                        # print("Object is not inside the patch")
                        continue
                    polygon = np.array(obj['polygon'])

                    
                    
                    # convert box to polygon
                    shape_polygon_patch = Polygon([[patch_box[1], patch_box[0]],
                                                   [patch_box[3], patch_box[0]],
                                                   [patch_box[3], patch_box[2]],
                                                   [patch_box[1], patch_box[2]]
                                                  ]
                                                 )
                    
                    # calculate the intersection area over the ground truth annotation
                    # check annotation poly is valid or not.
                    
                    
                    try:
                        shape_polygon_annotation = Polygon(polygon)
                        part_iou, inter_poly = calculate_intersection_over_poly1(shape_polygon_annotation,
                                                                                 shape_polygon_patch)
#                     except shapely.errors.TopologicalError, ValueError:
                    except:
                        part_iou, interpoly = 1, None                        
                    
                    # pdb.set_trace()
                    if part_iou < threshold_area:
                        continue
                    elif part_iou < 0.95:
                        polygon = convert_poly_2_numpy(inter_poly)
                    else:
                        polygon = polygon
                            
                    polygon = truncate_polygon(polygon, patch_box)
                    super_name = category_map[label]
                    super_id = name_to_idx_map[super_name]
                    object_list.append(super_id)
                    polygon_list.append(polygon)
                    
            return object_list, polygon_list



        # read ground truth list
        json_list = glob(os.path.join(root_dir, "*/*.json"))

        tif_list = glob(os.path.join(root_dir, "*/*.tif"))

        # create dictionary for ground true list
        json_dict = {json_file.split('/')[-2]: json_file for json_file in json_list}
        tif_dict = {tif_file.split('/')[-2]: tif_file for tif_file in tif_list}

        # read all the big images into memory
        image_dict = {tif_file.split('/')[-2]: cv2.imread(tif_file) for tif_file in tif_list} 
        for key in image_dict:
            print("Read the {} big image".format(key))
            image_dict[key] = cv2.cvtColor(image_dict[key], cv2.COLOR_BGR2RGB)
        # self.image_dict = {key: cv2.cvtColor(image_dict[key], cv2.COLOR_BGR2RGB) for key in image_dict}
        self.image_dict = image_dict

        # get GT dict
        gt_dict = dict()
        for key in json_dict:
            with open(json_dict[key]) as fid:
                gt_dict[key] = json.load(fid)


        # add center point for polygon
        for key in gt_dict:    # key: '001', '002'....
            # update object annotation with center coordination 
            for i, anno in enumerate(gt_dict[key]['objects']):
                # this happens as a references.
                anno['center'] = np.mean(anno['polygon'], axis=0)



        patch_height = 1024
        patch_width = 1024
        step_height = 500
        step_width = 500
        counter = 0
        for big_image_id in json_dict:
            # The shape of the big image
#             if big_image_id !=  '002':
#                  continue
            print("Processing: ", big_image_id)
            
            height, width = image_dict[big_image_id].shape[:2]
            annotation_dict = copy.deepcopy(gt_dict[big_image_id])

            for h in tqdm(range(0, height-patch_height, step_height)):
                for w in range(0, width-patch_width, step_width): 
                    # print("name_to_idx_map ", name_to_idx_map) 
                    object_list, polygon_list = get_class_id_and_polygon([h, w], 
                                                                         [patch_height, patch_width], 
                                                                         annotation_dict,
                                                                         category_map,
                                                                         name_to_idx_map
                                                                         )
                    if not object_list:
                        # continue skip the empty list image
                        continue
                        
                    # pdb.set_trace()
                    self.add_image(source="satellite",
                                   image_id=counter,
                                   path=None,
                                   width=patch_width,
                                   height=patch_height,
                                   object_list=object_list,
                                   polygon_list=polygon_list,
                                   big_image_id=big_image_id,
                                   start_h=h,
                                   start_w=w,
                                   image_type="big"
                                  )

                    counter += 1

# Configuration
class SatelliteConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overwrides values specific to the toy shapes dataset.
    """
    # Give the configuration a recognizable name.
    NAME = "satellite"

    # Train on 1 GPU and 8 images per GPU. We put multiple images on each GPU because the images are small.
    # BATCH_SIZE = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 4  # 4
    IMAGES_PER_GPU = 4

    # For 2019-01-11 
    # NUM_CLASSES = 1 + 69  # background + # classes
 
    NUM_CLASSES = 1 + 18  # background + # classes

    # For training on 001 only 
    # NUM_CLASSES = 1 + 30  # background + # classes
    # Use small images for fast traning. Set the limits of the small side and the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024
    # Use smaller anchors because our images and objects are small.
    # RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128, 256)
    RPN_ANCHOR_SCALES = (32, 64, 128, 256, 512)

    # Reduce ROIs per image because the images are small and have few objects. Aim to allow ROI sampling to pick 33%  positive ROIs.
    # TRAIN_ROIS_PER_IMAGE = 100

    # Use a small epoch since the data is simple.
    STEPS_PER_EPOCH = 300
    # Use a small validation steps since the epoch is small
    VALIDATION_STEPS = 5

class InferenceConfig(SatelliteConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


if __name__ == '__main__':
    
    """
    dataset_train = SatelliteDataset()
    dataset_train.config_dataset(dataset_name='satellite', 
                             json_file='/home/ubuntu/data/satellite/sate/annotations/instances_train2018.json',
                             skip_classes=['Road','Body_Of_water','Tree'],
                             root_dir='/home/ubuntu/data/satellite/sate')


    config = SatelliteConfig()
    config.display()
    with open('/home/ubuntu/data/complete_set/annotations/complete.json') as f:
        data_info = json.load(f)
    # id_name_map, static = get_static_info(data_info)
    duplicate_instances(data_info)
    category_map = category_map_name_to_super('../data/complete.json')
    print(category_map)
    print(len(category_map[0]))
    print(len(category_map[1]))
    # print(id_name_map, static)
    """

    dataset_train = SatelliteDataset()
    dataset_train.config_dataset_with_big_image(dataset_name='satellite',
                                                root_dir="../data/big_image")
    dataset_train.prepare()
    image_ids = np.arange(10)
        
    for idx in range(len(dataset_train.image_info)):
    #for idx in image_ids:
    
#         print("idx = {}".format(idx))
#         image = dataset_train.load_image(idx)
#         print("image.shape = {}".format(image.shape))
#         mask_array, class_id_array = dataset_train.load_mask(idx)
#         print("mask_array.shape = {}".format(mask_array.shape))
#         print("class_id.shape = {}".format(class_id_array.shape))
        dataset_train.visualize(idx)

