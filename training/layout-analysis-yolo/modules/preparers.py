import os
import json
import shutil
import random
import math

from PIL import Image


class DatasetCreator:
    def __init__(self,
                 raw_data_dir,
                 dataset_dir,
                 train_split=0.8):

        self.raw_data_dir = raw_data_dir
        self.dataset_dir = dataset_dir

        self.classes_path = os.path.join(self.raw_data_dir, 'classes.txt')
        self.json_min_path = os.path.join(self.raw_data_dir, 'labels.json')

        self.data_yaml_path = os.path.join(self.dataset_dir, 'data.yaml')

        self.images_dir = os.path.join(
            self.raw_data_dir, 'images')  # Raw images
        self.labels_dir = os.path.join(
            self.raw_data_dir, 'labels')  # Raw labels

        self.class_mapping = {"trash": 0,
                              "table": 1,
                              "image": 2}

        self.yolo_obb_creator = YoloOBBCreator(raw_data_dir=raw_data_dir,
                                               json_min_path=self.json_min_path,
                                               class_mapping=self.class_mapping)

        self.__train_folder = None
        self.__val_folder = None
        self.__test_folder = None

        self.__images_labels_dict = None

        self.train_split = train_split
        self.val_split = 0.6 * (1 - self.train_split)
        self.test_split = 1 - self.train_split - self.val_split

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

    def read_classes_file(self):
        with open(self.classes_path, 'r') as classes_file:
            # Set the names of the classes
            classes = [i.split('\n')[0] for i in classes_file.readlines()]
            classes = sorted(classes, key=lambda x: self.class_mapping[x])

        return classes

    def write_data_yaml(self):
        # Read classes file
        classes = self.read_classes_file()
        print(f"Available classes is {classes}")

        # Write the data.yaml file
        with open(self.data_yaml_path, 'w') as yaml_file:

            yaml_file.write('train: ' + self.train_folder + '\n')
            yaml_file.write('val: ' + self.val_folder + '\n')
            yaml_file.write('test: ' + self.test_folder + '\n')

            yaml_file.write('nc: ' + str(len(classes)) + '\n')

            yaml_file.write('names:' + '\n')
            for class_name in classes:
                yaml_file.write(
                    f"  {self.class_mapping.get(class_name, -1)}: {class_name}\n")

    def transform_and_save_image(self,
                                 image_name,
                                 copy_to):
        full_image_input_path = os.path.join(self.images_dir, image_name)
        new_image_name = os.path.splitext(image_name)[0] + '.jpg'
        full_image_output_path = os.path.join(copy_to, new_image_name)

        image = Image.open(full_image_input_path)
        image = image.convert("RGB")
        image.save(full_image_output_path, 'JPEG')
        image.close()

    def transform_and_save_files_from_dict(self,
                                           image_name,
                                           label_name,
                                           copy_to):

        self.transform_and_save_image(image_name, copy_to)

        if label_name is not None:
            shutil.copyfile(os.path.join(self.labels_dir, label_name),
                            os.path.join(copy_to, label_name))

    def partitionate_data(self):
        # Dict with images and labels
        data = self.images_labels_dict

        # Create the train, validation, and test datasets
        num_train = int(len(data) * self.train_split)
        num_val = int(len(data) * self.val_split)
        num_test = int(len(data) * self.test_split)

        train_data = {key: data[key] for key in list(data.keys())[:num_train]}
        val_data = {key: data[key] for key in list(
            data.keys())[num_train:num_train+num_val]}
        test_data = {key: data[key] for key in list(
            data.keys())[num_train+num_val:num_train+num_val+num_test]}

        # Copy the images and labels to the train, validation, and test folders
        for image_name, label_name in train_data.items():
            self.transform_and_save_files_from_dict(image_name=image_name,
                                                    label_name=label_name,
                                                    copy_to=self.train_folder)

        for image_name, label_name in val_data.items():
            self.transform_and_save_files_from_dict(image_name=image_name,
                                                    label_name=label_name,
                                                    copy_to=self.val_folder)

        for image_name, label_name in test_data.items():
            self.transform_and_save_files_from_dict(image_name=image_name,
                                                    label_name=label_name,
                                                    copy_to=self.test_folder)

    def process(self):
        # Creating labels from json
        print("Labels in obb format are creating...")
        self.yolo_obb_creator.create_yolo_bboxes()
        # Create train, valid, test datasets
        print("Dataset is creating...")
        self.partitionate_data()
        print("Train, validation, test datasets have created.")
        self.write_data_yaml()
        print("data.yaml file has created.")


class YoloOBBCreator():
    def __init__(self,
                 raw_data_dir,
                 json_min_path,
                 class_mapping):

        self.raw_data_dir = raw_data_dir
        self.labels_dir = os.path.join(raw_data_dir, 'labels')
        os.makedirs(self.labels_dir, exist_ok=True)

        self.json_min_path = json_min_path

        self.class_mapping = class_mapping

    def create_yolo_bboxes(self):
        '''
        Opens json file with annotation, takes labels, 
        format it to obb format and writes it to txt file
        '''
        # Open json file
        with open(self.json_min_path, 'r') as json_file:
            data = json.load(json_file)

        # Iterating through annotations
        for item in data:
            image_path = item["image"]
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            txt_filename = os.path.join(self.labels_dir, image_name + ".txt")

            # Iterating through labels in one annotation
            with open(txt_filename, 'w') as txt_file:
                if not 'label' in item:
                    pass

                else:
                    for label in item['label']:
                        class_label = label['rectanglelabels'][0]
                        class_index = self.class_mapping.get(class_label, -1)
                        if class_index == -1:
                            print(
                                f"There is no class label '{class_label}', change class mapping.")
                            continue

                        # Skip images and tables
                        elif class_index in [1, 2]:
                            continue

                        # Get points in obb format
                        points = self.get_rotated_rectangle(label['x'],
                                                            label['y'],
                                                            label['width'],
                                                            label['height'],
                                                            label['rotation'],
                                                            label['original_width'],
                                                            label['original_height'])

                        # Write labels in obb format to txt file
                        txt_file.write(f"{class_index} " + " ".join(
                            f"{coord[0]:.6f} {coord[1]:.6f}" for coord in points) + "\n")

    def get_rotated_rectangle(self,
                              x,
                              y,
                              w,
                              h,
                              theta,
                              original_width,
                              original_height):
        x1 = x / 100
        y1 = y / 100

        w = w * original_width
        h = h * original_height

        x2 = (x * original_width + w * math.cos(math.radians(theta))) / \
            original_width / 100
        y2 = (y * original_height + w * math.sin(math.radians(theta))) / \
            original_height / 100
        x3 = (x * original_width + w * math.cos(math.radians(theta)) - h * math.sin(
            math.radians(theta))) / original_width / 100
        y3 = (y * original_height + w * math.sin(math.radians(theta)) + h * math.cos(
            math.radians(theta))) / original_height / 100

        x4 = (x * original_width - h * math.sin(math.radians(theta))) / \
            original_width / 100
        y4 = (y * original_height + h * math.cos(math.radians(theta))) / \
            original_height / 100

        return [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]
