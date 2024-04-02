from labels import LSLabelFormatter

import os

json_input_path = "or_bb_labels.json"
json_output_path = ""

# label_formatter = LSLabelFormatter(json_input_path=json_input_path,
#                                    json_output_path=json_output_path)

label_formatter = LSLabelFormatter()

label_formatter.convert_or_bbox_to_polygon(json_input_path=json_input_path)
