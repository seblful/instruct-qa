from labels import LSLabelFormatter

import os

json_input_path = "inputs/or_bb_labels.json"
json_output_path = "outputs/polygon_labels_conv.json"


def main():
    label_formatter = LSLabelFormatter()

    label_formatter.convert_or_bbox_to_polygon(json_input_path=json_input_path,
                                               json_output_path=json_output_path)


if __name__ == "__main__":
    main()
