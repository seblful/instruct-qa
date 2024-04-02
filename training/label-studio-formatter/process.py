from labels import LSLabelFormatter

import os

json_input_path = "or_bb_labels.json"
json_output_path = "polygon_labels.json"


def main():
    label_formatter = LSLabelFormatter()

    label_formatter.convert_or_bbox_to_polygon(json_input_path=json_input_path)


if __name__ == "__main__":
    main()
