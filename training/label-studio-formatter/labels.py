from abc import ABC, abstractmethod
import os
import json
import copy
from PIL import Image, ImageDraw
import math
import numpy as np


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

    @abstractmethod
    def convert_to_relative(self,
                            image_width,
                            image_height):
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

    def convert_to_relative(self,
                            image_width,
                            image_height):
        x = self.x * image_width / 100
        y = self.y * image_height / 100
        width = self.width * image_width / 100
        height = self.height * image_height / 100

        return x, y, width, height


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

    def convert_to_relative(self,
                            image_width,
                            image_height):
        points = [(x * image_width / 100, y * image_height / 100)
                  for x, y in self.points]

        return points


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

    def transform_or_bbox_to_polygon(self,
                                     rect_label,
                                     image_width,
                                     image_height):
        # Get points
        x, y, w, h = rect_label.convert_to_relative(image_width,
                                                    image_height)
        rotation, label = rect_label.rotation, rect_label.label

        # Calculate angle and center
        angle = np.radians(rotation)
        center = np.array([x + w/2, y + h/2])

        # Calculate the coordinates of the four corners of the rectangle
        corners = np.array([[-w/2, -h/2],
                            [w/2, -h/2],
                            [w/2, h/2],
                            [-w/2, h/2]])

        # Rotation matrix
        rotation_matrix = np.array([[np.cos(angle), np.sin(angle)],
                                    [-np.sin(angle), np.cos(angle)]])

        # Rotate the corners around the center of the rectangle
        rotated_corners = np.dot(corners, rotation_matrix) + center

        # Format corners to absolute points
        rotated_corners = rotated_corners / \
            np.array([image_width, image_height]) * 100

        # Create PolygonLabel object
        polygon = PolygonLabel(points=rotated_corners.tolist(),
                               label=label)

        return polygon

    def fill_value(self,
                   polygon):

        value = dict()

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
                           polygons,
                           rect_labels):
        # Create an image with white background
        image = Image.open(image_path)
        image_width, image_height = image.size

        # Initialize the drawing context with the image as background
        draw = ImageDraw.Draw(image, 'RGBA')

        # Draw ellipse
        for r_label in rect_labels:
            # Get points
            x, y, w, h = r_label.convert_to_relative(image_width, image_height)

            # Find center of the polygon
            cx, cy = x + w/2, y + h/2

            # Define ellipse coordinates
            ellipse_width, ellipse_height = 50, 50
            left = cx - ellipse_width // 2
            top = cy - ellipse_height // 2
            right = cx + ellipse_width // 2
            bottom = cy + ellipse_height // 2

            draw.ellipse([left, top, right, bottom], fill='red')

        # Draw polygon
        for polygon in polygons:
            points = polygon.convert_to_relative(image_width, image_height)
            # Draw the polygon
            draw.polygon(points, fill=(0, 0, 255, 125))

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

            # Create list to store polygons
            polygons = []
            rect_labels = []

            # Process if result is not blank
            if result:
                # Iterating through results
                for res in result:
                    # Get value from result
                    image_width, image_height = res['original_width'], res['original_height']
                    value = res['value']
                    # Create RectangleLabel instance
                    rect_label = RectangleLabel(x=value['x'],
                                                y=value['y'],
                                                width=value['width'],
                                                height=value['height'],
                                                rotation=value['rotation'],
                                                label=value['rectanglelabels'][0])

                    # Transform oriented bbox to polygon and append it to polygons
                    polygon = self.transform_or_bbox_to_polygon(rect_label,
                                                                image_width,
                                                                image_height)

                    polygons.append(polygon)
                    rect_labels.append(rect_label)

                    # Fill value and result
                    value = self.fill_value(polygon)
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
                                        polygons=polygons,
                                        rect_labels=rect_labels)

            elif task["file_upload"] == "316e0c7c-20_10_3069_p_0.jpg":
                self.visualize_polygons(image_path="outputs/20_10_3069_p_0.jpg",
                                        polygons=polygons,
                                        rect_labels=rect_labels)

            elif task["file_upload"] == "45ee26df-10529_16_21_s_0.jpg":
                self.visualize_polygons(image_path="outputs/10529_16_21_s_0.jpg",
                                        polygons=polygons,
                                        rect_labels=rect_labels)

                # break

        # Write json to file
        self.write_json(json_output_path, json_output)
