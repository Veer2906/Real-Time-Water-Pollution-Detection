import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers

num_bounding_box_to_predict = 2
#image (720, 1280, 3)


def create_mlp_model(image_height, image_width, num_channels):
    model = keras.Sequential([
        layers.Flatten(input_shape=(image_height, image_width, num_channels)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(num_bounding_box_to_predict*4, activation='linear')
    ])

    return model

def load_data(folder_path):
        
    image_folder_path = folder_path+"images"
    annotation_folder_path = folder_path+"annotations"
    image_files = os.listdir(image_folder_path)
    num_images = min(len(image_files), 30)
    images = []

    for i in range(num_images):
        image_path = os.path.join((image_folder_path), image_files[i])
        img = cv2.imread(image_path)
        images.append(img)
        
    xml_files = [f for f in os.listdir(annotation_folder_path) if f.endswith('.xml')]

    bounding_boxes = []

    for xml_file in xml_files[0:30]:
        xml_path = os.path.join(annotation_folder_path, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        image_bounding_box = []

        for obj in root.findall(".//object[name='bottle']"):
            xmin = int(obj.find("bndbox/xmin").text)
            ymin = int(obj.find("bndbox/ymin").text)
            xmax = int(obj.find("bndbox/xmax").text)
            ymax = int(obj.find("bndbox/ymax").text)
            image_bounding_box = image_bounding_box + [xmin, ymin, xmax, ymax]

            if len(image_bounding_box) >= (num_bounding_box_to_predict*4):
                break

        if len(image_bounding_box) < (num_bounding_box_to_predict*4):
            image_bounding_box = image_bounding_box + [0, 0, 0, 0] * (num_bounding_box_to_predict - len(image_bounding_box)//4)

        
        bounding_boxes.append(image_bounding_box)
        return images, bounding_boxes

            
if __name__ == "__main__":
    training_folder_path = "C:\\Users\\veera\\Desktop\\Veer\\Coding\\Python\\Veritas\\FloW_IMG\\FloW_IMG\\training\\"
    training_images, training_bounding_boxes = load_data(training_folder_path)
    print(training_images)

    test_folder_path = "C:\\Users\\veera\\Desktop\\Veer\\Coding\\Python\\Veritas\\FloW_IMG\\FloW_IMG\\test\\"
    test_images, test_bounding_boxes = load_data(test_folder_path)
    print(test_images)

