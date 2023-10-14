import os
import cv2
import numpy as np
import xml.etree.ElementTree as ET
import tensorflow as tf 
from tensorflow import keras 
from tensorflow.keras import layers
from PIL import Image

num_bounding_box_to_predict = 4
#image (720, 1280, 3)
desired_shape = (720, 1280)


def create_mlp_model(image_height, image_width, num_channels):
    model = keras.Sequential([
        layers.Flatten(input_shape=(image_height, image_width, num_channels)),
        layers.Dense(64, activation='relu'),
        layers.Dense(32, activation='relu'),
        layers.Dense(num_bounding_box_to_predict*4, activation='linear')
    ])

    return model

def load_data(folder_path, num_images_to_load, num_bounding_box_to_predict, desired_shape):
    image_folder_path = os.path.join(folder_path, "images")
    annotation_folder_path = os.path.join(folder_path, "annotations")
    image_files = os.listdir(image_folder_path)
    num_images = min(len(image_files), num_images_to_load)

    images = []
    bounding_boxes = []

    for i in range(num_images):
        image_path = os.path.join(image_folder_path, image_files[i])
        img = cv2.imread(image_path)
        img = cv2.resize(img, desired_shape, interpolation=cv2.INTER_AREA)
        images.append(img)

        xml_file = image_files[i].replace(".jpg", ".xml")
        xml_path = os.path.join(annotation_folder_path, xml_file)
        tree = ET.parse(xml_path)
        root = tree.getroot()
        image_bounding_box = []

        for obj in root.findall(".//object[name='bottle']"):
            xmin = int(obj.find("bndbox/xmin").text)
            ymin = int(obj.find("bndbox/ymin").text)
            xmax = int(obj.find("bndbox/xmax").text)
            ymax = int(obj.find("bndbox/ymax").text)
            image_bounding_box.extend([xmin, ymin, xmax, ymax])

            if len(image_bounding_box) >= (num_bounding_box_to_predict * 4):
                break

        if len(image_bounding_box) < (num_bounding_box_to_predict * 4):
            image_bounding_box.extend([0, 0, 0, 0] * (num_bounding_box_to_predict - len(image_bounding_box) // 4))

        bounding_boxes.append(image_bounding_box)

    return np.array(images), np.array(bounding_boxes)
            
if __name__ == "__main__":
    training_folder_path = "C:\\Users\\veera\\Desktop\\Veer\\Coding\\Python\\Veritas\\FloW_IMG\\FloW_IMG\\training\\"
    training_images, training_bounding_boxes = load_data(training_folder_path, 100, num_bounding_box_to_predict, desired_shape)

    training_images = np.array(training_images)
    training_bounding_boxes = np.array(training_bounding_boxes)
    training_images = np.reshape(training_images, (-1, 720, 1280, 3))



    model = create_mlp_model(720, 1280, 3)
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(training_images, training_bounding_boxes, epochs=10, verbose=1)

    del training_images
    del training_bounding_boxes

    test_folder_path = "C:\\Users\\veera\\Desktop\\Veer\\Coding\\Python\\Veritas\\FloW_IMG\\FloW_IMG\\test\\"
    test_images, test_bounding_boxes = load_data(test_folder_path, 100, num_bounding_box_to_predict, desired_shape)
    test_images = np.array(test_images)

    training_bounding_boxes = np.array(training_bounding_boxes)
    test_images = np.reshape(test_images, (-1, 720, 1280, 3))

    evaluation = model.evaluate(test_images, test_bounding_boxes)
    print(evaluation)

    #def load_data(folder_path, num_images_to_load, num_bounding_box_to_predict, desired_shape):

