import os
import json


class Label:
    pass


class PolygonLabel:
    pass


class BrushLabel:
    pass


class LSLabelFormatter:
    def __init__(self,
                 json_input_path,
                 json_output_path):

        json_input_dict = self.read_json(json_input_path)

    def read_json(self, json_input_path):

        with open(json_input_path) as json_file_input:
            json_dict = json.load(json_file_input)

        return json_dict

    def write_json(json_output_path, json_dict):
        with open(json_output_path, "w") as json_file_output:
            json.dump(json_dict, json_file_output)

        return None

    def convert_bb_to_polygon(self):
        # Extract the necessary values from the bounding box label
        x = bbox_label['value']['x']
        y = bbox_label['value']['y']
        width = bbox_label['value']['width']
        height = bbox_label['value']['height']
        label = bbox_label['value']['rectanglelabels'][0]

        # Calculate the coordinates of the four corners of the bounding box
        top_left = [x, y]
        top_right = [x + width, y]
        bottom_right = [x + width, y + height]
        bottom_left = [x, y + height]

        # Create the polygon label format
        polygon_label = {
            "original_width": bbox_label['original_width'],
            "original_height": bbox_label['original_height'],
            "image_rotation": bbox_label['image_rotation'],
            "value": {
                # Closed polygon
                "points": [top_left, top_right, bottom_right, bottom_left, top_left],
                "polygonlabels": [label]
            }
        }

        return polygon_label
