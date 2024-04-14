import os
import json
import random

import numpy as np
from PIL import Image
import cv2


class PolygonLabel():
    def __init__(self,
                 points,
                 label):
        self.points = points
        self.label = label

    def __repr__(self):
        return f"PolygonLabel: points={self.points}, label='{self.label}'"

    def convert_to_relative(self,
                            image_width,
                            image_height):
        points = [(x * image_width / 100, y * image_height / 100)
                  for x, y in self.points]

        return points


class DatasetCreator():
    def __init__(self,
                 raw_data_dir,
                 images_dir,
                 dataset_dir,
                 train_split=0.8) -> None:
        # Data dirs
        self.inputs_dir = raw_data_dir
        self.images_dir = images_dir
        self.labels_dir = os.path.join(raw_data_dir, 'labels')
        self.dataset_dir = dataset_dir

        # Input dirs
        self.json_polygon_path = os.path.join(
            raw_data_dir, 'polygon_labels.json')
        self.classes_path = os.path.join(raw_data_dir, 'classes.txt')

        # Data split
        self.train_split = train_split
        self.val_split = 0.6 * (1 - self.train_split)
        self.test_split = 1 - self.train_split - self.val_split

        self.__id2label = None
        self.__label2id = None

        self.__images_labels_dict = None

        # Dataset foldets
        self.__train_folder = None
        self.__val_folder = None
        self.__test_folder = None

    @property
    def train_folder(self):
        if self.__train_folder is None:
            train_folder = os.path.join(self.dataset_dir, 'train')
            # Creating folder for train set
            os.makedirs(train_folder, exist_ok=True)
            self.__train_folder = train_folder

        return self.__train_folder

    @property
    def val_folder(self):
        if self.__val_folder is None:
            val_folder = os.path.join(self.dataset_dir, 'val')
            # Creating folder for val set
            os.makedirs(val_folder, exist_ok=True)
            self.__val_folder = val_folder

        return self.__val_folder

    @property
    def test_folder(self):
        if self.__test_folder is None:
            test_folder = os.path.join(self.dataset_dir, 'test')
            # Creating folder for test set
            os.makedirs(test_folder, exist_ok=True)
            self.__test_folder = test_folder

        return self.__test_folder

    @property
    def images_labels_dict(self):
        '''
        Dict with names of images with corresponding 
        names of label
        '''
        if self.__images_labels_dict is None:
            self.__images_labels_dict = self.__create_images_labels_dict()

        return self.__images_labels_dict

    def __create_images_labels_dict(self, shuffle=True):
        # List of all images and labels in directory
        images = os.listdir(self.images_dir)
        # labels = os.listdir(self.labels_dir)

        # Create a dictionary to store the images and labels names
        images_labels = {}
        for image in images:
            label = image.rstrip('.jpg') + '.txt'

            images_labels[image] = label

        if shuffle:
            # Shuffle the data
            keys = list(images_labels.keys())
            random.shuffle(keys)
            images_labels = {key: images_labels[key] for key in keys}

        return images_labels

    @property
    def id2label(self):
        if self.__id2label is None:
            self.__id2label = self.__create_id2label()

        return self.__id2label

    @property
    def label2id(self):
        if self.__label2id is None:
            self.__label2id = {v: k for k, v in self.id2label.items()}

        return self.__label2id

    def __create_id2label(self):
        with open(self.classes_path, 'r') as classes_file:
            # Set the names of the classes
            classes = [i.split('\n')[0] for i in classes_file.readlines()]
            id2label = {k: v for k, v in enumerate(classes)}

        return id2label

    def _read_json(self, json_input_path):

        with open(json_input_path) as json_file_input:
            json_dict = json.load(json_file_input)

        return json_dict

    def __get_polygons(self, task):
        # Retrieve annotations and results
        annotation = task['annotations'][0]
        result = annotation['result']

        # Create list to store polygons and labels
        polygons = []

        # Process if result is not blank
        if result:
            # Get image width and image height
            image_width, image_height = result[0]['original_width'], result[0]['original_height']
            # Iterating through results
            for res in result:
                # Get value from result
                value = res['value']

                # Get polygon and label
                polygon = PolygonLabel(points=value['points'],
                                       label=value['polygonlabels'][0])

                # Append labels to lists
                polygons.append(polygon)

        # Sort polygons
        polygons.sort(key=lambda x: self.label2id[x.label], reverse=True)

        return polygons, image_width, image_height

    def __convert_polygons_to_mask(self,
                                   polygons,
                                   image_width,
                                   image_height):
        # Generate mask
        mask = np.zeros((image_height, image_width), dtype=np.int32)

        # Iterating through polygons
        for polygon in polygons:
            # Converts points to relative
            relative_points = polygon.convert_to_relative(
                image_width, image_height)
            relative_points = np.array(relative_points, dtype=np.int32)

            # Fill mask
            mask = cv2.fillPoly(
                mask, [relative_points], self.label2id[polygon.label])

        # Convert mask to uint8
        mask = mask.astype(np.uint8)

        return mask

    def __save_mask(self,
                    mask_array,
                    image_name):
        # Create mask path
        mask_name = os.path.splitext(image_name)[0] + '.png'
        mask_path = os.path.join(self.labels_dir, mask_name)

        # Convert mask array to image and save it
        mask_image = Image.fromarray(mask_array)
        mask_image.save(mask_path)

        return None

    def _create_mask_images(self):
        # Read json_polygon_path
        json_input = self._read_json(self.json_polygon_path)

        # Iterating through tasks
        for task in json_input:
            # Get polygons and labels
            polygons, image_width, image_height = self.__get_polygons(task)

            # Get image path and mask path
            image_name = os.path.basename(task["data"]["image"]).split('-')[1]

            # Create mask
            mask_array = self.__convert_polygons_to_mask(polygons=polygons,
                                                         image_width=image_width,
                                                         image_height=image_height)

            # Save mask as images
            self.__save_mask(mask_array=mask_array,
                             image_name=image_name)
