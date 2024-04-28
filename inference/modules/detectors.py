import os

import numpy as np
from PIL import Image
import cv2

import torch
import ultralytics

from modules.instructors import Instruction


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

    def predict(self, image):
        results = self.model(image)

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
                 model_path):
        self.model_path = model_path


class ImageProcessor:
    def clean_image(self, image):
        pass

    def clean_roi_of_image(self):
        pass

    def process(self, image):
        pass


class TesseractOCR:
    def __init__(self):
        self.ocr_model = ''


class InstructionProcessor:
    def __init__(self,
                 instr_dir,
                 yolo_stamp_det_model_path,
                 segformer_la_model_path):
        # Paths
        self.instr_dir = instr_dir

        # Detection, LA, OCR models
        self.yolo_stamp_det = YOLOStampDetector(model_path=yolo_stamp_det_model_path,
                                                model_type='n')

        self.segformer_la = SegformerLayoutAnalyser(
            model_path=segformer_la_model_path)

        self.tesseract_ocr = TesseractOCR()

        # Image processor
        self.image_processor = ImageProcessor()

    def extract_text(self, instruction):
        # Check if input is Instruction instance
        assert isinstance(instruction, Instruction)

        # Iterating through images in instruction
        for image in instruction.instr_imgs:
            image_array = np.array(image)
            results = self.yolo_stamp_det.predict(image)

            # Get bboxrs
            bboxes = results[0].obb.xyxyxyxy.detach(
            ).cpu().numpy().astype(np.int32)
            # Iterating through bbox
            for bbox in bboxes:
                print(bbox)

            break
