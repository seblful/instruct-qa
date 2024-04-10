from abc import ABC, abstractmethod
import sys
import os

import copy
import json
import math
from PIL import Image, ImageDraw

import numpy as np
import cv2


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

    def convert_to_polygon_label(self,
                                 image_width,
                                 image_height):
        # Get points
        x, y, w, h = self.convert_to_relative(image_width,
                                              image_height)

        # Calculate angle and center and bias center
        angle = np.radians(self.rotation)
        d_rotation = (self.rotation + 180) % 360 - 180
        bias_coef = np.array(
            [-0.048 * image_width, 0.09 * image_height]) * np.radians(d_rotation)
        center = np.array([x + w/2, y + h/2])
        biased_center = center + bias_coef

        # Calculate the coordinates of the four corners of the rectangle
        corners = np.array([[-w/2, -h/2],
                            [w/2, -h/2],
                            [w/2, h/2],
                            [-w/2, h/2]])

        # Rotation matrix
        rotation_matrix = np.array([[np.cos(angle), np.sin(angle)],
                                    [-np.sin(angle), np.cos(angle)]])

        # Rotate the corners around the center of the rectangle
        rotated_corners = np.dot(corners, rotation_matrix) + biased_center

        # Format corners to absolute points
        rotated_corners = rotated_corners / \
            np.array([image_width, image_height]) * 100

        # # Add bias to polygon
        # d_rotation = (rotation + 180) % 360 - 180
        # bias_coef = np.array([-0.10, 0.16]) * d_rotation
        # rotated_corners += bias_coef

        # Create PolygonLabel object
        polygon = PolygonLabel(points=rotated_corners.tolist(),
                               label=self.label)

        return polygon, biased_center


class PolygonLabel(Label):
    def __init__(self,
                 points,
                 label):
        self.points = points
        self.label = label

        self.__tuple_points = None

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

    def base_rle_encode(self, inarray):
        """run length encoding. Partial credit to R rle function.
        Multi datatype arrays catered for including non Numpy
        returns: tuple (runlengths, startpositions, values)"""
        ia = np.asarray(inarray)  # force numpy
        n = len(ia)
        if n == 0:
            return None, None, None
        else:
            y = ia[1:] != ia[:-1]  # pairwise unequal (string safe)
            i = np.append(np.where(y), n - 1)  # must include last element posi
            z = np.diff(np.append(-1, i))  # run lengths
            p = np.cumsum(np.append(0, z))[:-1]  # positions
            return z, p, ia[i]

    def bits2byte(self,
                  arr_str,
                  n=8):
        """Convert bits back to byte

        :param arr_str:  string with the bit array
        :type arr_str: str
        :param n: number of bits to separate the arr string into
        :type n: int
        :return rle:
        :type rle: list
        """
        rle = []
        numbers = [arr_str[i: i + n] for i in range(0, len(arr_str), n)]
        for i in numbers:
            rle.append(int(i, 2))
        return rle

    def encode_rle(self,
                   arr,
                   wordsize=8,
                   rle_sizes=[3, 4, 8, 16]):
        """Encode a 1d array to rle

        :param arr: flattened np.array from a 4d image (R, G, B, alpha)
        :type arr: np.array
        :param wordsize: wordsize bits for decoding, default is 8
        :type wordsize: int
        :param rle_sizes:  list of ints which state how long a series is of the same number
        :type rle_sizes: list
        :return rle: run length encoded array
        :type rle: list
        """
        # Set length of array in 32 bits
        num = len(arr)
        numbits = f'{num:032b}'

        # put in the wordsize in bits
        wordsizebits = f'{wordsize - 1:05b}'

        # put rle sizes in the bits
        rle_bits = ''.join([f'{x - 1:04b}' for x in rle_sizes])

        # combine it into base string
        base_str = numbits + wordsizebits + rle_bits

        # start with creating the rle bite string
        out_str = ''
        for length_reeks, p, value in zip(*self.base_rle_encode(arr)):
            # TODO: A nice to have but --> this can be optimized but works
            if length_reeks == 1:
                # we state with the first 0 that it has a length of 1
                out_str += '0'
                # We state now the index on the rle sizes
                out_str += '00'

                # the rle size value is 0 for an individual number
                out_str += '000'

                # put the value in a 8 bit string
                out_str += f'{value:08b}'
                state = 'single_val'

            elif length_reeks > 1:
                state = 'series'
                # rle size = 3
                if length_reeks <= 8:
                    # Starting with a 1 indicates that we have started a series
                    out_str += '1'

                    # index in rle size arr
                    out_str += '00'

                    # length of array to bits
                    out_str += f'{length_reeks - 1:03b}'

                    out_str += f'{value:08b}'

                # rle size = 4
                elif 8 < length_reeks <= 16:
                    # Starting with a 1 indicates that we have started a series
                    out_str += '1'
                    out_str += '01'

                    # length of array to bits
                    out_str += f'{length_reeks - 1:04b}'

                    out_str += f'{value:08b}'

                # rle size = 8
                elif 16 < length_reeks <= 256:
                    # Starting with a 1 indicates that we have started a series
                    out_str += '1'

                    out_str += '10'

                    # length of array to bits
                    out_str += f'{length_reeks - 1:08b}'

                    out_str += f'{value:08b}'

                # rle size = 16 or longer
                else:
                    length_temp = length_reeks
                    while length_temp > 2**16:
                        # Starting with a 1 indicates that we have started a series
                        out_str += '1'

                        out_str += '11'
                        out_str += f'{2 ** 16 - 1:016b}'

                        out_str += f'{value:08b}'
                        length_temp -= 2**16

                    # Starting with a 1 indicates that we have started a series
                    out_str += '1'

                    out_str += '11'
                    # length of array to bits
                    out_str += f'{length_temp - 1:016b}'

                    out_str += f'{value:08b}'

        # make sure that we have an 8 fold lenght otherwise add 0's at the end
        nzfill = 8 - len(base_str + out_str) % 8
        total_str = base_str + out_str
        total_str = total_str + nzfill * '0'

        rle = self.bits2byte(total_str)

        return rle

    def convert_to_brush_label(self,
                               image_width,
                               image_height):
        # Converts points to relative
        relative_points = self.convert_to_relative(image_width, image_height)
        relative_points = np.array(relative_points, dtype=np.int32)

        # Generate mask from polygons and preprocess it
        mask = np.zeros((image_height, image_width), dtype=np.int32)
        mask = cv2.fillPoly(mask, [relative_points], 255)
        mask = mask.astype(np.uint8)
        array = np.repeat(mask.ravel(), 4)

        # Convert to rle
        rle = self.encode_rle(array)
        brush_label = BrushLabel(points=rle,
                                 mask=mask,
                                 label=self.label)

        return brush_label


class BrushLabel(Label):
    def __init__(self,
                 points,
                 mask,
                 label):
        self.points = points
        self.mask = mask
        self.label = label

        self.__tuple_points = None

    def __repr__(self):
        return f"BrushLabel: points={self.points}, label='{self.label}'"

    @property
    def tuple_points(self):
        if self.__tuple_points is None:
            self.__tuple_points = [tuple(points) for points in self.points]

        return self.__tuple_points

    def convert_to_relative(self):
        points = [x / 255 for x in self.points]

        return points


class LabelsVisualizer():
    def __init__(self,
                 images_dir):
        self.images_dir = images_dir
        self.images_listdir = [image for image in os.listdir(
            images_dir) if image.endswith(('.jpg', '.png'))]

    def check_image_exists(self,
                           image_path):
        if not image_path in self.images_listdir:
            print(f"Image {image_path} not in images directory.")
            return False

        return True

    def visualize_polygons(self,
                           image_path,
                           polygons,
                           biased_centers,
                           image_width,
                           image_height):

        # Check if image in the folder
        if not self.check_image_exists(image_path):
            return

        # Create full path and open image
        full_image_path = os.path.join(self.images_dir, image_path)
        image = Image.open(full_image_path)

        # Initialize the drawing context with the image as background
        draw = ImageDraw.Draw(image, 'RGBA')

        # Draw ellipse
        for biased_center in biased_centers:
            # Find center of the polygon
            cx, cy = biased_center

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

    def visualize_brushes(self,
                          image_path,
                          brushes,
                          color=[255, 0, 0]):

        # Check if image in the folder
        if not self.check_image_exists(image_path):
            return

        # Create full path and open image
        full_image_path = os.path.join(self.images_dir, image_path)
        image = Image.open(full_image_path)
        image = np.array(image)

        # Create copy of image
        masked_image = image.copy()

        # Iterating through brushes and draw mask
        for brush in brushes:
            mask = brush.mask

            # Apply the mask to the image
            masked_image = np.where(np.repeat(mask[:, :, np.newaxis], 3, axis=2),
                                    np.array(color, dtype='uint8'),
                                    masked_image)
            masked_image = masked_image.astype(np.uint8)

        # addWeighted
        masked_image = cv2.addWeighted(image, 0.3, masked_image, 0.7, 0)

        # Convert and show image
        image = Image.fromarray(masked_image)
        image.show()


class LSLabelFormatter:
    def __init__(self,
                 images_dir=None):
        self.labels_type_translator = {"rectangle": "rectanglelabels",
                                       "polygon": "polygonlabels",
                                       "brush": "brushlabels"}

        self.labels_translator = {"table": "Table",
                                  "image": "Picture",
                                  "trash": "Stamp"}

        self.from_name_translator = {'rectangle': 'label',
                                     'polygon': 'label',
                                     'brush': 'tag'}

        self.visualizer = LabelsVisualizer(images_dir=images_dir)

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

    def fill_value(self,
                   label_to,
                   label):

        # Create empty dict
        value = dict()

        # Fill by new values
        if label_to == "polygon":
            value['points'] = label.points
            value['closed'] = True
            value['polygonlabels'] = [self.labels_translator[label.label]]

        elif label_to == "brush":
            value['format'] = "rle"
            value['rle'] = label.points
            value['brushlabels'] = [label.label]

        return value

    def fill_result(self,
                    label_to,
                    result,
                    value):
        # Make a copy of result
        result = copy.deepcopy(result)

        # Fill by new values
        result['id'] = result['id'] + "_N"
        result['value'] = value
        result['type'] = self.labels_type_translator[label_to]
        result['from_name'] = self.from_name_translator[label_to]
        result['to_name'] = 'image'

        return result

    def convert_labels(self,
                       label_from,
                       label_to,
                       json_input_path,
                       json_output_path,
                       visualize=False):

        # Read json input
        json_input = self.read_json(json_input_path)

        # Create blank list for dicts
        json_output = []

        # Iterating through tasks
        for task in json_input:
            # Retrieve annotations and results
            annotation = task['annotations'][0]
            result = annotation['result']

            # Create blank list to store result
            new_result = []

            # Create list to store labels
            labels_input = []
            labels_output = []
            biased_centers = []

            # Process if result is not blank
            if result:
                # Iterating through results
                for res in result:
                    # Get value from result
                    image_width, image_height = res['original_width'], res['original_height']
                    value = res['value']

                    if label_from == "rectangle":
                        # Create RectangleLabel instance
                        label_input = RectangleLabel(x=value['x'],
                                                     y=value['y'],
                                                     width=value['width'],
                                                     height=value['height'],
                                                     rotation=value['rotation'],
                                                     label=value[self.labels_type_translator[label_from]][0])

                        # Transform oriented bbox to polygon and append it to polygons
                        label_output, biased_center = label_input.convert_to_polygon_label(image_width,
                                                                                           image_height)

                        biased_centers.append(biased_center)

                    elif label_from == "polygon":
                        # Create PolygonLabel instance
                        label_input = PolygonLabel(points=value['points'],
                                                   label=value[self.labels_type_translator[label_from]][0])

                        # Transform polygon to brush
                        label_output = label_input.convert_to_brush_label(image_width,
                                                                          image_height)

                    # Append labels to lists
                    labels_input.append(label_input)
                    labels_output.append(label_output)

                    # Fill value and result
                    value = self.fill_value(label_to=label_to,
                                            label=label_output)
                    res = self.fill_result(label_to=label_to,
                                           result=res,
                                           value=value)

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

            # Visualize polygons
            if visualize:
                # Retrieve image path
                image_path = os.path.basename(
                    task["data"]["image"]).split('-')[1]

                if label_to == "polygon":
                    # Visualizie polygon labels
                    self.visualizer.visualize_polygons(image_path=image_path,
                                                       polygons=labels_output,
                                                       biased_centers=biased_centers,
                                                       image_width=image_width,
                                                       image_height=image_height)

                elif label_to == "brush":
                    self.visualizer.visualize_brushes(image_path=image_path,
                                                      brushes=labels_output)

        # Write json to file
        self.write_json(json_output_path, json_output)
