import os

from labels import LSLabelFormatter

# Setup paths
HOME = os.getcwd()
# Directory with images
INSTRUCT_QA_DIR = os.path.dirname(os.path.dirname(HOME))
LS_INPUT_DIR = os.path.join(INSTRUCT_QA_DIR, 'data', 'ls-input-data')

# JSON paths to convert from oriented bboxes to polygons
json_or_bb_in_path = "inputs/or_bb_labels.json"
json_polygons_in_path = "inputs/polygon_labels.json"

# JSON paths to convert from polygons to brush labels
json_polygons_out_path = "outputs/polygon_labels_converted.json"
json_brush_out_path = "outputs/brush_labels_converted.json"


def main():
    # Create LSLabelFormatter instance
    label_formatter = LSLabelFormatter(images_dir=LS_INPUT_DIR)

    # # Convert from oriented bboxes to polygons
    # label_formatter.convert_labels(label_from="rectangle",
    #                                label_to="polygon",
    #                                json_input_path=json_or_bb_in_path,
    #                                json_output_path=json_polygons_out_path,
    #                                visualize=True)

    # Convert from polygons to brush labels
    label_formatter.convert_labels(label_from="polygon",
                                   label_to="brush",
                                   json_input_path=json_polygons_in_path,
                                   json_output_path=json_brush_out_path,
                                   visualize=False)


if __name__ == "__main__":
    main()
