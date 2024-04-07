from labels import LSLabelFormatter

import os

json_or_bb_in_path = "inputs/or_bb_labels.json"
json_polygons_in_path = "inputs/polygon_labels.json"

json_polygons_out_path = "outputs/polygon_labels_converted.json"
json_brush_out_path = "outputs/brush_labels_converted.json"


def main():
    label_formatter = LSLabelFormatter()

    # label_formatter.convert_labels(label_from="rectangle",
    #                                label_to="polygon",
    #                                json_input_path=json_or_bb_in_path,
    #                                json_output_path=json_polygons_out_path,
    #                                visualize=True)

    label_formatter.convert_labels(label_from="polygon",
                                   label_to="brush",
                                   json_input_path=json_polygons_in_path,
                                   json_output_path=json_brush_out_path,
                                   visualize=False)


if __name__ == "__main__":
    main()
