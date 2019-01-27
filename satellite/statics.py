"""
Get a statics of training data.
"""
import sys

import json

# json_file = "/home/ubuntu/data/complete_set/annotations/train.json"
json_file = "../data/complete.json"

with open(json_file) as f:
    data_info = json.load(f)


print(data_info['categories'])
fid = open("class_name.txt", 'w')
for category in data_info['categories']:
    print("id: {:5},\t name: {:25},\tsupercategory: {:25}".format(category['id'], category['name'], category['supercategory']))
    fid.write("id: {:5},\t name: {:25},\tsupercategory: {:25}\n".format(category['id'], category['name'], category['supercategory']))
fid.close()
sys.exit()

print("instance number = {:5d}".format(len(data_info['annotations']))) 
print("image number = {:5d}".format(len(data_info['images']))) 
print("category number = {:5d}".format(len(data_info['categories']))) 


# Get instance number for each category

static = dict()

id_name_map = dict()

for cls in data_info['categories']:
    if cls['id'] not in id_name_map:
        id_name_map[cls['id']] = cls['name']

print(id_name_map)

for ins in data_info['annotations']:
    cls_name = id_name_map[ins['category_id']]
    if cls_name not in static:
        static[cls_name] = 0
    else:
        static[cls_name] += 1
print(len(static))
miss_category =[]
with open("instance_statics_skip_0_instance.txt", "w") as fid:
    for key in static:
        if static[key] == 0:
            continue
        fid.write("{:25}\t{}\n".format(key+":", static[key]))


for idx in id_name_map:
    miss_flag = True
    for cls in static:
        if id_name_map[idx] == cls:
            # find the class
            miss_flag = False
            break
    if miss_flag:
        miss_category.append(id_name_map[idx])

print(miss_category)
    
