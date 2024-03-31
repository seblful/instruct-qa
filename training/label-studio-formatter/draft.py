from labels import LSLabelFormatter

import os

json_input_path = "polygon_labels.json"
json_output_path = ""

label_formatter = LSLabelFormatter(json_input_path=json_input_path,
                                   json_output_path=json_output_path)
