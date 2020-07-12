#!/usr/bin/env python3

import datetime
import json
import os
import re
import fnmatch
from PIL import Image
import cv2
import numpy as np
from pycococreatortools import pycococreatortools

# 这里设置一些文件路径
ROOT_DIR = '/home/loke/samples/train'  # 根目录
IMAGE_DIR = os.path.join(ROOT_DIR, "left")  # 根目录下存放你原图的文件夹
ANNOTATION_DIR = os.path.join(ROOT_DIR, "gt")  # 根目录下存放mask标签的文件夹

# 这里就是填一些有关你数据集的信息
INFO = {
    "description": "Example Dataset",
    "url": "https://github.com/waspinator/pycococreator",
    "version": "0.1.0",
    "year": 2018,
    "contributor": "waspinator",
    "date_created": datetime.datetime.utcnow().isoformat(' ')
}

LICENSES = [
    {
        "id": 1,
        "name": "Attribution-NonCommercial-ShareAlike License",
        "url": "http://creativecommons.org/licenses/by-nc-sa/2.0/"
    }
]

# 这里是你数据集的类别，这里有三个分类，就是square, circle, triangle。制作自己的数据集主要改这里就行了
CATEGORIES = [
    {
        'id': 1,
        'name': 'change',
        'supercategory': 'shape',
    },

]
class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32,
                              np.float64)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def filter_for_jpeg(root, files):
    file_types = ['*.jpeg', '*.jpg','*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]

    return files


def filter_for_annotations(root, files, image_filename):
    file_types = ['*.png']
    file_types = r'|'.join([fnmatch.translate(x) for x in file_types])
    basename_no_extension = os.path.splitext(os.path.basename(image_filename))[0]
    file_name_prefix = basename_no_extension + '.*'
    files = [os.path.join(root, f) for f in files]
    files = [f for f in files if re.match(file_types, f)]
    files = [f for f in files if re.match(file_name_prefix, os.path.splitext(os.path.basename(f))[0])]

    return files

def prepare_cd_data_from_txt(path):
    train_left_input_names=[]
    train_right_input_names = []
    train_output_names=[]



    for index, line in enumerate(open(path, 'r',encoding='gbk')):
        name = line.split()
        if(len(name) == 2):
            train_left_input_names.append(name[0])
            train_right_input_names.append(name[1])
        elif(len(name) == 1 and name[0] != '\n'):
            train_output_names.append(name[0])
        elif (len(name) == 3):
            train_left_input_names.append(name[0])
            train_right_input_names.append(name[1])
            train_output_names.append(name[2])

    return train_left_input_names,train_right_input_names,train_output_names

def prepare_cd_bounding_from_txt(path):
    train_bounding_names = []
    val_bounding_names = []
    boundings = []
    for index, line in enumerate(open(path, 'r', encoding='gbk')):
        # if index != 0 and line.count(':')>0:
        if len(boundings) > 0 and line.count(':') > 0:
            train_bounding_names.append(boundings)
            boundings = []
        else:
            coordinates = []
            coord = line.split()
            if len(coord) == 4:
                coordinates.append(int(coord[0]))
                coordinates.append(int(coord[1]))
                coordinates.append(int(coord[2]))
                coordinates.append(int(coord[3]))
                boundings.append(coordinates)


    return train_bounding_names

def generateJson(image_files,annotation_files,boundings,path):
    coco_output = {
        "info": INFO,
        "licenses": LICENSES,
        "categories": CATEGORIES,
        "images": [],
        "annotations": []
    }

    image_id = 1
    segmentation_id = 1


    # go through each image
    for i in range(len(image_files)):

        if np.array(boundings[i]).sum() == 0:
            continue

        image = Image.open(image_files[i])
        image_info = pycococreatortools.create_image_info(
            image_id, image_files[i], image.size)
        coco_output["images"].append(image_info)



        annotation_filename = annotation_files[i]

        # go through each associated annotation
        #print(annotation_filename)
        for j in range(len(boundings[i])):
            img = cv2.imread(annotation_filename)
            img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
            binary_mask = np.zeros((np.shape(img)[0],np.shape(img)[1]),np.uint8)

            y1 = boundings[i][j][0]
            x1 = boundings[i][j][1]
            y2 = boundings[i][j][2]
            x2 = boundings[i][j][3]

            if y1 == y2 and x1 == x2:
                continue

            binary_mask[x1:x2,y1:y2] = img[x1:x2,y1:y2]


            binary_mask = np.uint8(binary_mask > 0)

            class_id = np.max(img)
            if class_id == 255:
                class_id = 1

            category_info = {'id': class_id, 'is_crowd': 0}


            annotation_info = pycococreatortools.create_annotation_info(
                    segmentation_id, image_id, category_info, binary_mask,
                    image.size, tolerance=2)

            if annotation_info is not None:
                coco_output["annotations"].append(annotation_info)

            '''else:
                annotation_info = {
                    "id": segmentation_id,
                    "image_id": image_id,
                    "category_id": category_info["id"],
                    "iscrowd": category_info["is_crowd"],
                    "area": 0,
                    "bbox": [],
                    "segmentation": [],
                    "width": binary_mask.shape[1],
                    "height": binary_mask.shape[0],
                }
                coco_output["annotations"].append(annotation_info)'''

            segmentation_id = segmentation_id + 1


        image_id = image_id + 1


    with open(path.format(ROOT_DIR), 'w') as output_json_file:
        json.dump(coco_output, output_json_file, cls=NumpyEncoder)



def main():


    path = "/home/loke/samples/val.txt"

    left_image_files,right_image_files,gt_image_file=prepare_cd_data_from_txt(path)

    bounding_path = "/home/loke/samples/val_bounding.txt"
    boundings = prepare_cd_bounding_from_txt(bounding_path)

    generateJson(left_image_files,gt_image_file,boundings,"/home/loke/samples/val_left.json")






if __name__ == "__main__":
    main()
