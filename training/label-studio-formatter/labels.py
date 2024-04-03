import os
import json
import copy
from PIL import Image, ImageDraw
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
            self.__tuple_points = [tuple(points) for points in self.points]

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

    def write_json(self,
                   json_output_path,
                   json_dict):
        with open(json_output_path, "w") as json_file_output:
            json.dump(json_dict, json_file_output)

        return None

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

        # corners = np.array([[-w/2, h/2],
        #                     [w/2, h/2],
        #                     [w/2, -h/2],
        #                     [-w/2, -h/2]])

        corners = np.array([[-w/2, -h/2],
                            [w/2, -h/2],
                            [w/2, h/2],
                            [-w/2, h/2]])

        # Rotation matrix
        rotation_matrix = np.array([[np.cos(angle), np.sin(angle)],
                                    [-np.sin(angle), np.cos(angle)]])

        # Rotate the corners around the center of the rectangle
        center = np.array([x + w/2, y + h/2])
        rotated_corners = np.dot(corners, rotation_matrix) + center

        # Create PolygonLabel object
        polygon = PolygonLabel(points=rotated_corners.tolist(),
                               label=label)

        return polygon

    import math

    # def obb_to_polygon(x, y, w, h, rotation):
    #     # Convert degrees to radians for rotation
    #     angle_rad = math.radians(rotation)

    #     # Calculate the half-diagonal
    #     half_diagonal = math.sqrt(w**2 + h**2) / 2

    #     # Calculate the angle to the corner
    #     theta = math.atan2(h, w)

    #     # Calculate the offset for each corner
    #     offset_dx = half_diagonal * math.cos(angle_rad + theta)
    #     offset_dy = half_diagonal * math.sin(angle_rad + theta)

    #     # Calculate the coordinates for each corner
    #     corner_1 = (x - offset_dx, y - offset_dy)
    #     corner_2 = (x + offset_dy, y - offset_dx)
    #     corner_3 = (x + offset_dx, y + offset_dy)
    #     corner_4 = (x - offset_dy, y + offset_dx)

    #     # Return the polygon as a list of points
    #     return [corner_1, corner_2, corner_3, corner_4]

    def fill_value(self,
                   value,
                   polygon):

        # Make a copy of value
        value = copy.deepcopy(value)

        # Clear value
        value.clear()

        # Fill by new values
        value['points'] = polygon.points
        value['closed'] = True
        value['polygonlabels'] = [polygon.label]

        return value

    def fill_result(self,
                    result,
                    value):
        # Make a copy of result
        result = copy.deepcopy(result)

        # Fill by new values
        result['id'] = result['id'] + "_N"
        result['value'] = value
        result['type'] = 'polygonlabels'

        return result

    def visualize_polygons(self,
                           image_path,
                           polygon):
        # Create an image with white background
        image = Image.open(image_path)
        width, height = image.size

        # Initialize the drawing context with the image as background
        draw = ImageDraw.Draw(image, 'RGBA')

        # Format points to relative coordinates
        points = [(x * width / 100, y * height / 100)
                  for x, y in polygon.points]

        # Draw the polygon
        draw.polygon(points, fill=(255, 0, 0, 125))

        # Display the image
        image.show()

    def convert_or_bbox_to_polygon(self,
                                   json_input_path,
                                   json_output_path):

        # Read json input
        json_input = self.read_json(json_input_path)

        # Create blank list for dicts
        json_output = []

        # Iterating through tasks
        for i, task in enumerate(json_input):
            # Retrieve annotations and results
            annotation = task['annotations'][0]
            result = annotation['result']

            # Create blank list to store result
            new_result = []

            # Process if result is not blank
            if result:
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

                    # Transform oriented bbox to polygon and append it to polygons
                    polygon = self.transform_or_bbox_to_polygon(rect_label)

                    # Fill value and result
                    value = self.fill_value(value, polygon)
                    res = self.fill_result(res, value)

                    # Append res to new result
                    new_result.append(res)

            else:
                # Paste empty result
                pass

            # Edit annotation and task and append it to json output
            annotation = copy.deepcopy(annotation)
            task = copy.deepcopy(task)
            annotation['result'] = new_result
            task['annotations'] = [annotation]
            json_output.append(task)

            if task["file_upload"] == "b9fe78e4-16_12_730_p_0.jpg":
                self.visualize_polygons(image_path="outputs/16_12_730_p_0.jpg",
                                        polygon=polygon)

        # Write json to file
        self.write_json(json_output_path, json_output)
