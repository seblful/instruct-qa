import os
import json

import math
import numpy as np

from abc import ABC, abstractmethod


class Label(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __repr__(self):
        pass

    @property
    @abstractmethod
    def tuple_points(self):
        pass


class RectangleLabel(Label):
    def __init__(self,
                 x,
                 y,
                 width,
                 height,
                 rotation,
                 label):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.rotation = rotation
        self.label = label

        self.__tuple_points = None

    def __repr__(self) -> str:
        return f"RectangleLabel: x={self.x}, y={self.y}, width={self.width}, height={self.height}, label='{self.label}'"

    @property
    def tuple_points(self):
        if self.__tuple_points is None:
            self.__tuple_points = (
                self.x, self.y, self.width, self.height, self.rotation, self.label)

        return self.__tuple_points


class PolygonLabel:
    def __init__(self,
                 points,
                 label):
        self.points = points
        self.label = label

        self.__tuple_points = None
        pass

    def __repr__(self):
        return f"PolygonLabel: points={self.points}, label='{self.label}'"

    @property
    def tuple_points(self):
        if self.__tuple_points is None:
            self.__tuple_points = tuple([tuple(points)
                                        for points in self.points])

        return self.__tuple_points


class BrushLabel:
    pass


class LSLabelFormatter:
    def __init__(self):

        # json_input_dict = self.read_json(json_input_path)
        pass

    def read_json(self, json_input_path):

        with open(json_input_path) as json_file_input:
            json_dict = json.load(json_file_input)

        return json_dict

    def write_json(json_output_path, json_dict):
        with open(json_output_path, "w") as json_file_output:
            json.dump(json_dict, json_file_output)

        return None

    def get_ids(self, task):
        task_id = task['id']
        annotation_id = task['annotations'][0]['id']
        by_id = task['annotations'][0]['completed_by']

        return task_id, annotation_id, by_id

    def get_image_size(self, result):
        image_width = result[0]['original_width']
        image_height = result[0]['original_height']

        return image_width, image_height

    def transform_or_bbox_to_polygon(self, rect_label):
        # Get points
        x, y, w, h, rotation, label = rect_label.tuple_points
        # Calculate the coordinates of the four corners of the rectangle

        # Calculate angle and corners
        angle = np.radians(rotation)
        corners = np.array([[-w/2, -h/2],
                            [w/2, -h/2],
                            [w/2, h/2],
                            [-w/2, h/2]])

        # Rotation matrix
        rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                    [np.sin(angle), np.cos(angle)]])

        # Rotate the corners around the center of the rectangle
        center = np.array([x + w/2, y + h/2])
        rotated_corners = np.dot(corners, rotation_matrix.T) + center

        # Create PolygonLabel object
        polygon = PolygonLabel(points=rotated_corners.tolist(),
                               label=label)

        return polygon

    def convert_or_bbox_to_polygon(self,
                                   json_input_path):

        # Read json input
        json_input = self.read_json(json_input_path)

        # Create blank list for dicts
        json_output = []

        # Iterating through tasks
        for index, task in enumerate(json_input):
            # Retrieve results
            result = task['annotations'][0]['result']

            # Process if result is not blank
            if result:
                # Retrieve ids
                task_id, annotation_id, by_id = self.get_ids(task)

                # Iterating through results
                for res in result:
                    # Get value from result
                    value = res['value']
                    # Create RectangleLabel instance
                    rect_label = RectangleLabel(x=value['x'],
                                                y=value['y'],
                                                width=value['width'],
                                                height=value['height'],
                                                rotation=value['rotation'],
                                                label=value['rectanglelabels'][0])

                    polygon = self.transform_or_bbox_to_polygon(rect_label)
                    print(polygon)

                # results = self.fill_results(task, polygons, name_of_object)

                break
