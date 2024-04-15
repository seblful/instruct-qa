import os
from PIL import Image

from transformers import SegformerImageProcessor


class SegFormerDataset():
    def __init__(self,
                 set_dir,
                 checkpoint="vikp/surya_det2"):
        # Dirs, paths with images, masks and classes
        self.set_dir = set_dir
        self.images_dir = os.path.join(set_dir, 'images')
        self.masks_dir = os.path.join(set_dir, 'masks')
        self.classes_path = os.path.join(set_dir, os.pardir, 'classes.txt')

        # Feauture extractor
        self.processor = SegformerImageProcessor.from_pretrained(checkpoint)

        # Create list if images and labels names
        self.images_listdir = [image for image in os.listdir(
            self.images_dir) if image.endswith('jpg')]
        self.masks_listdir = [label for label in os.listdir(
            self.masks_dir) if label.endswith('.png')]
        # Assert if number of images and masks is the same
        assert len(self.images_listdir) == len(self.masks_listdir)

        # id and label
        self.__id2label = None
        self.__label2id = None

    @property
    def id2label(self):
        if self.__id2label is None:
            # Open classes_path and extract names from there
            with open(self.classes_path, 'r') as classes_file:
                classes = [i.split('\n')[0] for i in classes_file.readlines()]
                id2label = {k: v for k, v in enumerate(classes)}

            self.__id2label = id2label

        return self.__id2label

    @property
    def label2id(self):
        if self.__label2id is None:
            self.__label2id = {v: k for k, v in self.id2label.items()}

        return self.__label2id

    def __len__(self):
        return len(self.images_listdir)

    def __getitem__(self, idx):
        # Open image and mask
        image = Image.open(os.path.join(self.root_dir, self.images[idx]))
        segmentation_map = Image.open(
            os.path.join(self.set_dir, self.masks_listdir[idx]))

        # Randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.processor(
            image, segmentation_map, return_tensors="pt")

        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension

        return encoded_inputs
