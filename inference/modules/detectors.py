import os
import re

import numpy as np
from PIL import Image
import cv2

import torch
import torch.nn.functional as F
import ultralytics
from transformers import SegformerConfig, SegformerImageProcessor

import pytesseract

from modules.instructors import Instruction
from modules.segformer_model import SegformerForRegressionMask


class YOLOStampDetector:
    def __init__(self,
                 model_path,
                 model_type='n'):

        # Dict to translate indexes to classes
        self.ind2classes = {0: 'trash'}

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model_path = model_path
        self.model_yaml = f"yolov8{model_type}-obb.yaml"
        self.model_type = f"yolov8{model_type}-obb.pt"

        self.__model = None

    @property
    def model(self):
        if self.__model == None:
            # # Build a new model from scratch
            # model = ultralytics.YOLO(self.model_yaml)
            # Load a pretrained model
            model = ultralytics.YOLO(self.model_type)

            # Load from my pretrained model
            model = ultralytics.YOLO(self.model_path)

            # Transfer model to device
            model = model.to(self.device)

            self.__model = model

        return self.__model

    def predict(self, image, verbose=False):
        results = self.model(image, verbose=verbose)

        return results

    def visualize_detection(self, results):
        # plot a BGR numpy array of predictions
        image_array = results[0].plot(line_width=5)
        image = Image.fromarray(image_array[..., ::-1])  # RGB PIL image
        image.show()  # show image

    def crop_keys(self, image, results):
        """
        Process the bounding boxes produced by the table detection model into
        cropped table images and cropped tokens.
        """
        # Create dict to store cropped parts of image
        crops = []

        # Convert image to array
        image_array = np.array(image)

        # Extract predicted classes, confidednces, obbs
        cls = results[0].obb.cls.detach().cpu().numpy()
        # conf = results[0].obb.conf.detach().cpu().numpy()
        xyxyxyxy = results[0].obb.xyxyxyxy.detach().cpu().numpy()

        # Crop every part of image and add it to dict
        for i in range(len(cls)):
            cropped_image = self.crop_obb(image=image_array,
                                          points=xyxyxyxy[i])

            crops.append(cropped_image)

        return crops

    def crop_obb(self, image, points):
        # Find the minimum area rotated rectangle
        rect = cv2.minAreaRect(points)

        # Get width and height of the detected rectangle
        width = int(rect[1][0])
        height = int(rect[1][1])

        # Represent the corners of the rectangle.
        box = cv2.boxPoints(rect)

        # Coordinate of the points in box points after the rectangle has been straightened
        dst_pts = np.array([[0, height-1],
                            [0, 0],
                            [width-1, 0],
                            [width-1, height-1]], dtype="float32")

        # The perspective transformation matrix
        M = cv2.getPerspectiveTransform(box, dst_pts)

        # Directly warp the rotated rectangle to get the straightened rectangle
        warped = cv2.warpPerspective(image, M, (width, height))

        # cv2.imwrite("crop_img.jpg", warped)

        return warped


class SegformerLayoutAnalyser:
    def __init__(self,
                 model_path,
                 config_path):
        # Paths
        self.model_path = model_path

        # Device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Config, id2label, label2id
        self.config = SegformerConfig.from_pretrained(config_path)
        self.id2label = self.config.id2label
        self.label2id = {v: k for k, v in self.id2label.items()}

        # Model and processor
        self.__model = None
        self.__processor = None

        # Color map
        self.color_map = {0: (0, 0, 0),
                          1: (255, 0, 0),
                          2: (0, 255, 0),
                          3: (0, 0, 255),
                          4: (255, 255, 0),
                          5: (255, 0, 255),
                          6: (0, 255, 255),
                          7: (128, 128, 0)}

    @property
    def model(self):
        if self.__model is None:
            # Load model and checkpoint
            model = SegformerForRegressionMask(config=self.config)
            checkpoint = torch.load(self.model_path, map_location=self.device)

            # Create a new state dictionary without the "model." prefix
            new_state_dict = {}

            for key, value in checkpoint['state_dict'].items():
                new_key = key.removeprefix("model.")
                new_state_dict[new_key] = value

            # Load state dict
            model.load_state_dict(new_state_dict)

            # Move model to device
            model = model.to(self.device)

            # Turn model to evaluation
            model.eval()

            self.__model = model

        return self.__model

    @property
    def processor(self):
        if self.__processor is None:
            processor = SegformerImageProcessor.from_pretrained(
                "vikp/surya_layout", do_reduce_labels=False)
            size = self.model.config.image_size
            processor.size = {"height": size, "width": size}

            self.__processor = processor

        return self.__processor

    def preprocess_image(self, image_array):
        # Add dimension if image is gray
        if len(image_array.shape) == 2:
            image_array = np.dstack([image_array, image_array, image_array])

        image_tensor = self.processor(image_array, return_tensors="pt")
        image_tensor = image_tensor['pixel_values'].to(self.device)

        return image_tensor

    def predict(self, image_array):
        # Preprocess image
        image_tensor = self.preprocess_image(image_array)

        # Get predictions
        with torch.no_grad():
            results = self.model(image_tensor)

        upsampled_logits = F.interpolate(
            results.logits, size=image_array.shape[0:2], mode="bilinear", align_corners=False)
        pred_mask = upsampled_logits.argmax(dim=1)[0].cpu().numpy()

        return pred_mask

    def visualize_mask(self,
                       image_array,
                       pred_mask):
        # Convert image to 3d
        if len(image_array.shape) == 2:
            image_array = np.dstack([image_array, image_array, image_array])

        # Fill mask with colors
        color_mask = np.zeros(pred_mask.shape + (3,))
        for i, _ in self.color_map.items():
            color_mask[pred_mask == i] = self.color_map[i]
        color_mask = color_mask.astype(np.uint8)

        # Blend image with mask
        blend_image = cv2.addWeighted(image_array, 0.5, color_mask, 0.5, 0)
        blend_image = Image.fromarray(blend_image)

        # Show image
        blend_image.show()

        return blend_image


class ImageProcessor:
    def __init__(self,
                 yolo_stamp_det_model_path,
                 segformer_la_model_path,
                 segformer_la_config_path,
                 surya_model_list):

        # Detection, LA models
        self.yolo_stamp_det = YOLOStampDetector(model_path=yolo_stamp_det_model_path,
                                                model_type='n')

        self.segformer_la = SegformerLayoutAnalyser(model_path=segformer_la_model_path,
                                                    config_path=segformer_la_config_path)

        # Define the target classes you want to extract
        self.target_classes = [1, 2, 3, 4]

        # Surya model list
        self.surya_model_list = surya_model_list

    def convert_image_to_3d(self,
                            image_array):
        if len(image_array.shape) == 2:
            image_array = np.dstack([image_array, image_array, image_array])

        return image_array

    def clean_whole_image(self,
                          image_array,
                          alpha=1,
                          beta=0):
        converted_image = cv2.convertScaleAbs(image_array,
                                              alpha=alpha,
                                              beta=beta)

        return converted_image

    def detect_stamps(self,
                      image):
        # Predict stamps with yolo
        results = self.yolo_stamp_det.predict(image)
        bboxes = results[0].obb.xyxyxyxy.detach(
        ).cpu().numpy().astype(np.int32)

        return bboxes

    def analyze_layout(self,
                       image_array):
        # Layout analysis
        la_mask = self.segformer_la.predict(
            image_array=image_array)

        return la_mask

    def clean_roi_of_image(self,
                           image_array,
                           bboxes,
                           alpha=3,
                           beta=0):

        # Create a mask for the bounding box region
        mask = np.zeros_like(image_array)

        # Iterating through bboxes and clean_roi of bbox
        for bbox in bboxes:
            # Fill polygon
            cv2.fillPoly(mask, [bbox], (255, 255, 255))

            # Extract the region of interest (ROI) from the image
            roi = cv2.bitwise_and(image_array, mask)

            # Apply cv2.convertScaleAbs to the ROI
            converted_image = cv2.convertScaleAbs(roi,
                                                  alpha=alpha,
                                                  beta=beta)

            # Update the original image with the processed ROI
            converted_image = np.where(
                mask == 255, converted_image, image_array)

        return converted_image

    def apply_la_mask(self,
                      image_array,
                      la_mask):

        # # Visualize layout analysis prediction
        # self.segformer_la.visualize_mask(image_array=image_array,
        #                                  pred_mask=pred_la_mask)

        # Create a white background image
        white_background = np.ones_like(image_array) * 255

        # Create a mask for the target classes
        target_mask = np.isin(la_mask, self.target_classes).astype(np.uint8)

        # Extract the relevant parts from the original image
        masked = cv2.bitwise_and(
            image_array, image_array, mask=target_mask)

        # Paste the extracted parts onto the white background
        cleaned_img = cv2.bitwise_and(
            white_background, masked, white_background, mask=target_mask)

        return cleaned_img

    def process(self, image):
        # Create image array and copy of it
        image_array = np.array(image)

        # Convert image array to 3d
        image_array = self.convert_image_to_3d(image_array=image_array)

        # Layout analysis
        la_mask = self.analyze_layout(image_array=image_array)

        # Detect stamp
        bboxes = self.detect_stamps(image=image)

        # Extract subset of cleaned image defined by mask from layout analysis
        cleaned_img = self.apply_la_mask(image_array=image_array,
                                         la_mask=la_mask)

        # Clean roi
        cleaned_img = self.clean_roi_of_image(image_array=cleaned_img,
                                              bboxes=bboxes,
                                              alpha=4)

        # Clean the whole image
        cleaned_img = self.clean_whole_image(image_array=cleaned_img,
                                             alpha=2)

        return Image.fromarray(cleaned_img)
