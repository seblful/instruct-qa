import os
from PIL import Image

from modules.instructors import Instruction
from modules.detectors import YOLOStampDetector


# Setup paths
HOME = os.getcwd()

DATA_DIR = os.path.abspath(os.path.join(os.pardir, 'data'))
INSTR_DIR = os.path.join(DATA_DIR, 'instructions')

MODELS_DIR = os.path.join(HOME, "models")
YOLO_MODEL_PATH = os.path.join(MODELS_DIR, "yolo_best.pt")


def main():
    pdf_path = os.path.join(INSTR_DIR, '21_07_3165_i.pdf')
    pdf_url = 'https://www.rceth.by/NDfiles/instr/21_07_3165_i.pdf'
    instruction = Instruction(instr_dir=INSTR_DIR,
                              pdf_url=pdf_url)  # , pdf_path=pdf_path)

    image = instruction.instr_imgs[1]

    # YOLO
    yolo_detector = YOLOStampDetector(model_path=YOLO_MODEL_PATH,
                                      model_type='n')

    results = yolo_detector.predict(image)

    print(results[0])

    crops = yolo_detector.crop_keys(image=image,
                                    results=results)
    cropped_table = Image.fromarray(crops['trash'])
    print(crops['trash'])
    cropped_table.show()
    # yolo_detector.visualize_detection(results)


if __name__ == '__main__':
    main()
