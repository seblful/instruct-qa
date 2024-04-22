from modules.instructors import Instruction, InstructsOCR

import os

DATA_DIR = os.path.abspath(os.path.join(os.pardir, 'data'))
INSTR_DIR = os.path.join(DATA_DIR, 'instructions')


def main():
    pdf_path = os.path.join(INSTR_DIR, '21_07_3165_i.pdf')
    pdf_url = 'https://www.rceth.by/NDfiles/instr/21_07_3165_i.pdf'
    instruction = Instruction(instr_dir=INSTR_DIR,
                              pdf_url=pdf_url)  # , pdf_path=pdf_path)

    image = instruction.instr_imgs[0]
    image.show()


if __name__ == '__main__':
    main()
