import os

from modules.instructors import Instruction
from modules.detectors import InstructionProcessor


# Setup paths
HOME = os.getcwd()

DATA_DIR = os.path.abspath(os.path.join(os.pardir, 'data'))
INSTR_DIR = os.path.join(DATA_DIR, 'instructions')

# Models
MODELS_DIR = os.path.join(HOME, "models")
YOLO_STAMP_DET_MODEL_PATH = os.path.join(MODELS_DIR, "yolo_stamp_det.pt")
SEGFORMER_LA_MODEL_PATH = os.path.join(MODELS_DIR, "segformer_la.ckpt")
SEGFORMER_LA_CONFIG_PATH = os.path.join(MODELS_DIR, "segformer_la_config.json")


def main():
    pdf_path = os.path.join(INSTR_DIR, '21_07_3165_i.pdf')
    pdf_url = "https://rceth.by/NDfiles/instr/24_04_2905_s.pdf"
    instruction = Instruction(instr_dir=INSTR_DIR,
                              pdf_url=pdf_url)  # , pdf_path=pdf_path)

    # Create InstructionProcessor instance
    instr_processor = InstructionProcessor(instr_dir=INSTR_DIR,
                                           yolo_stamp_det_model_path=YOLO_STAMP_DET_MODEL_PATH,
                                           segformer_la_model_path=SEGFORMER_LA_MODEL_PATH,
                                           segformer_la_config_path=SEGFORMER_LA_CONFIG_PATH)

    # Extract tect from instruction
    instr_processor.extract_text(instruction=instruction)


if __name__ == '__main__':
    main()
