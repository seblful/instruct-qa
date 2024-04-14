import os
import random


class DatasetCreator():
    def __init__(self,
                 raw_data_dir,
                 images_dir,
                 dataset_dir,
                 train_split=0.8) -> None:
        # Data dirs
        self.inputs_dir = raw_data_dir
        self.images_dir = images_dir
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
