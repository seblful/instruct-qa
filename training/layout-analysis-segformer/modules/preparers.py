import os
import shutil
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
        # Raw data dirs
        self.inputs_dir = raw_data_dir
        self.images_dir = images_dir
        self.masks_dir = os.path.join(raw_data_dir, 'masks')

        # Dataset dirs
        self.dataset_dir = dataset_dir
        self.__train_dir = None
        self.__val_dir = None
        self.__test_dir = None

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

    @property
    def train_dir(self):
        if self.__train_dir is None:
            train_dir = os.path.join(self.dataset_dir, 'train')
            images_dir = os.path.join(train_dir, 'images')
            masks_dir = os.path.join(train_dir, 'masks')
            # Creating folder for train set
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)

            self.__train_dir = train_dir

        return self.__train_dir

    @property
    def val_dir(self):
        if self.__val_dir is None:
            val_dir = os.path.join(self.dataset_dir, 'val')
            images_dir = os.path.join(val_dir, 'images')
            masks_dir = os.path.join(val_dir, 'masks')
            # Creating folder for val set
            os.makedirs(val_dir, exist_ok=True)
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)

            self.__val_dir = val_dir

        return self.__val_dir

    @property
    def test_dir(self):
        if self.__test_dir is None:
            test_dir = os.path.join(self.dataset_dir, 'test')
            images_dir = os.path.join(test_dir, 'images')
            masks_dir = os.path.join(test_dir, 'masks')
            # Creating folder for test set
            os.makedirs(test_dir, exist_ok=True)
            os.makedirs(images_dir, exist_ok=True)
            os.makedirs(masks_dir, exist_ok=True)

            self.__test_dir = test_dir

        return self.__test_dir

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
        # images = os.listdir(self.images_dir)
        labels = os.listdir(self.masks_dir)

        # Create a dictionary to store the images and labels names
        images_labels = {}
        for label in labels:
            image = os.path.splitext(label)[0] + '.jpg'

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
        mask_path = os.path.join(self.masks_dir, mask_name)

        # Convert mask array to image and save it
        mask_image = Image.fromarray(mask_array)
        mask_image.save(mask_path)

        return None

    def _create_mask_images(self):
        # Read json_polygon_path
        json_input = self._read_json(self.json_polygon_path)

        # Iterating through tasks
        for task in json_input:

            # # Take only completed results
            # if len(task['annotations'][0]['result']) <= 4:
            #     continue

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

    def __transform_and_save_image(self,
                                   image_name,
                                   copy_to):
        # Define image path
        full_image_input_path = os.path.join(self.images_dir, image_name)
        new_image_name = os.path.splitext(image_name)[0] + '.jpg'
        full_image_output_path = os.path.join(copy_to, new_image_name)

        # Convert image to jpg
        image = Image.open(full_image_input_path)
        image = image.convert("RGB")
        image.save(full_image_output_path, 'JPEG')
        image.close()

    def __transform_and_save_files_from_dict(self,
                                             image_name,
                                             label_name,
                                             copy_to):

        # Copy images
        images_dir = os.path.join(copy_to, 'images')
        self.__transform_and_save_image(image_name, images_dir)

        # Copy maskss
        if label_name is not None:
            masks_dir = os.path.join(copy_to, 'masks')
            shutil.copyfile(os.path.join(self.masks_dir, label_name),
                            os.path.join(masks_dir, label_name))

    def _partitionate_data(self):
        # Dict with images and labels
        data = self.images_labels_dict

        # Create the train, validation, and test datasets
        num_train = int(len(data) * self.train_split)
        num_val = int(len(data) * self.val_split)
        num_test = int(len(data) * self.test_split)

        # Create dicts with images and labels names
        train_data = {key: data[key] for key in list(data.keys())[:num_train]}
        val_data = {key: data[key] for key in list(
            data.keys())[num_train:num_train+num_val]}
        test_data = {key: data[key] for key in list(
            data.keys())[num_train+num_val:num_train+num_val+num_test]}

        # Copy the images and labels to the train, validation, and test folders
        for data_dict, folder_name in zip((train_data, val_data, test_data), (self.train_dir, self.val_dir, self.test_dir)):
            for image_name, label_name in data_dict.items():
                self.__transform_and_save_files_from_dict(image_name=image_name,
                                                          label_name=label_name,
                                                          copy_to=folder_name)

        # Copy classes.txt file
        shutil.copyfile(self.classes_path, os.path.join(
            self.dataset_dir, 'classes.txt'))

    def process(self):
        # Creating masks from polygon json
        print("Masks from polygons are creating...")
        self._create_mask_images()
        # Create train, valid, test datasets
        print("Dataset is creating...")
        self._partitionate_data()
        print("Train, validation, test datasets have created.")
