import os
import sys
import shutil

import xml.etree.ElementTree as ET
import os
import random as rd 
from math import ceil
import matplotlib.pyplot as plt

from random import shuffle, randint

#convert a xml label to a kitti label 
#output_folder should exist
#file path should end in a .xml
def convert_annotation(file_path, output_folder):
    xml_label = open(file_path)
    tree = ET.parse(xml_label)
    xml_root = tree.getroot()

    kitti_label_name = os.path.basename(file_path[:-4]) + ".txt"

    with open(os.path.join(output_folder, kitti_label_name), "w+") as f:
        for obj in xml_root.iter('object'):
            cls = obj.find('name').text
            cls = "".join(cls.split())
            xmlbox = obj.find('bndbox')
            """
                4 numbers from 5 to 8: the 2-dimensional bounding box of the object
                xmin，ymin，xmax，ymax
            """
            xmin, ymin, xmax, ymax = xmlbox.find('xmin').text, xmlbox.find('ymin').text, \
                                     xmlbox.find('xmax').text, xmlbox.find('ymax').text
            f.write(cls + " " + '0.0' + " " + '0' + " " + '1.0' + " " + str(xmin) + '.0' + " "
                    + str(ymin) + '.0' + " " + str(xmax) + '.0' + " " + str(ymax) + '.0' + " " +
                    str((int(str(ymax)) - int(str(ymin)))/int(1000) )+ " " + str((int(str(xmax)) - int(str(xmin)) )/int(1000))+ " " + '0.1' + " " + '1.0' + " " + '0.0' + " " + '1.0' + " " + '0.0' + '\n')  


#Create a subset of a dataset given a list of names, original datset path and the output folder path
#output folder should exist
def create_subset(original, name_list, output_folder):
    
    #determine image ext
    ext = os.path.splitext(os.listdir(os.path.join(original, "images"))[0])[1]
    
    
    image_out = os.path.join(output_folder, "images")
    label_out = os.path.join(output_folder, "labels")
    
    os.makedirs(image_out, exist_ok=True)
    os.makedirs(label_out, exist_ok=True)
    
    with open(name_list, "r") as ls:
        for line in ls:
            line = line.strip()
            shutil.copy(os.path.join(original,"images", line + ext), os.path.join(image_out, line + ext))
            shutil.copy(os.path.join(original, "labels", line + ".txt"), os.path.join(label_out, line + ".txt"))	
            
            
            
            
def combine_kitti(dir1, dir2, output):

    outputImages = os.path.join(output, "images")
    outputLabels = os.path.join(output, "labels")

    os.makedirs(outputImages,exist_ok = True)
    os.makedirs(outputLabels, exist_ok = True)

    folders = os.listdir(dir1)
    images1 = os.path.join(dir1, [x for x in folders if "image" in x][0])
    labels1 = os.path.join(dir1, [x for x in folders if "label" in x][0])

    folders = os.listdir(dir2)
    images2 = os.path.join(dir2, [x for x in folders if "image" in x][0])
    labels2 = os.path.join(dir2, [x for x in folders if "label" in x][0])


    count = 0
    for file in os.listdir(images1):

        filePath = os.path.join(images1, file)
        
        fileType = os.path.splitext(file)[1] #get extension
        shutil.copy(filePath, os.path.join(outputImages, str(count).zfill(5) + fileType))

        labelName = os.path.splitext(os.path.basename(filePath))[0]
        labelPath = os.path.join(labels1, labelName + ".txt")
        shutil.copy(labelPath, os.path.join(outputLabels, str(count).zfill(5) + ".txt"))

        print("Original Image: " + filePath + " New Image: " + os.path.join(outputImages, str(count).zfill(5) + fileType))


        count+=1


    for file in os.listdir(images2):

        filePath = os.path.join(images2, file)

        fileType = os.path.splitext(file)[1] #get extension
        shutil.copy(filePath, os.path.join(outputImages,str(count).zfill(5) + fileType))

        labelName = os.path.splitext(os.path.basename(filePath))[0]
        labelPath = os.path.join(labels2, labelName + ".txt")
        shutil.copy(labelPath, os.path.join(outputLabels, str(count).zfill(5) + ".txt"))

        print("Original Image: " + filePath + " New Image: " + os.path.join(outputImages, str(count).zfill(5) + fileType))


        count+=1

    print("Done combining")



def gen_random_aug_spec(img_x, img_y, ext,output):

    #boolean choice on rotation, translate, color, blur 

    enable_rotation = rd.choice([True, False])
    enable_translation = rd.choice([True, False])
    enable_shear = rd.choice([True, False])
    enable_color = rd.choice([True, False])
    enable_brightness = rd.choice([True, False])
    enable_blur = rd.choice([True, False])
    
    
    rot_angle = 0
    translate_x = 0
    translate_y = 0
    shear_ratio_x = 0
    shear_ratio_y = 0
    hue_rot_angle = 0
    sat_shift = 1
    brightness_offset = 0
    contrast = 0
    center = 127.5
    blur_size = 0
    blur_std = 1
    
    if(enable_rotation):
        rot_angle = rd.randint(0,180)
   
    if(enable_translation):
        translate_x = rd.randint(0, img_x/4)
        translate_y = rd.randint(0, img_y/4)

    if(enable_shear):
        shear_ratio_x = rd.uniform(0.0, 0.5) #not sure on upper bound that is reasonable
        shear_ratio_y = rd.uniform(0.0, 0.5)
        
        
    if(enable_color):
        hue_rot_angle = rd.randint(0,359)
        sat_shift = rd.uniform(0.3,1) #not sure on good lower bound, 1 is unchanged
        
    if(enable_brightness):
        brightness_offset = rd.randint(0, 75) #too high and the images are just white
        #contrast and center seem to mess up the image to much if the value isn't in a good range
        #contrast = rd.uniform(0,1) #0 is unchanged, not sure on upper bound 
        #center = rd.uniform(100, 200) #127.5 is common
        
    if(enable_blur):
        blur_size = rd.randrange(1,20,2) #may want a smaller upper bound - only odd for some reason
        blur_std = rd.uniform(0.8, 1.2) #not sure on bound 
        

    flipV = rd.choice([True, False])
    flipH = rd.choice([True, False])


    aug_spec = f"""
    spatial_config{{

      rotation_config{{
        angle: {rot_angle}
        units: "degrees"
      }}
      
      flip_config{{
        flip_vertical: {flipV}
        flip_horizontal: {flipH}
      }}
      
      shear_config{{
        shear_ratio_x: {shear_ratio_x}
        shear_ratio_y: {shear_ratio_y}
      }}
      
      translation_config{{
        translate_x: {translate_x}
        translate_y: {translate_y}
      }}
    }}

    color_config{{

      hue_saturation_config{{
        hue_rotation_angle: {hue_rot_angle}
        saturation_shift: {sat_shift}
      }}
      
      contrast_config{{
        contrast: {contrast}
        center: {center}
      
      }}
      
      brightness_config{{
        offset: {brightness_offset}
      }}
      
    }}

    blur_config {{
     std: {blur_std}
     size: {blur_size}
    }}

    # Setting up dataset config.
    dataset_config{{
      image_path: "images"
      label_path: "labels"
    }}

    output_image_width: {img_x}
    output_image_height: {img_y}
    output_image_channel: 3
    image_extension: "{ext}"
    """

    with open(output, "w+") as f:
        f.write(aug_spec)



def visualize_images(image_dir, num_cols=4, num_images=10):
    valid_image_ext = ['.jpg', '.png', '.jpeg', '.ppm']
    output_path =  image_dir
    num_rows = int(ceil(float(num_images) / float(num_cols)))
    f, axarr = plt.subplots(num_rows, num_cols, figsize=[80,30])
    f.tight_layout()
    
    images = os.listdir(image_dir)
    shuffle(images)
    for idx, img_path in enumerate(images[:num_images]):
        col_id = idx % num_cols
        row_id = idx // num_cols
        img = plt.imread(os.path.join(image_dir,img_path))
        axarr[row_id, col_id].imshow(img) 