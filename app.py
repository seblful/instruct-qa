from modules.instructors import Instructs

import os

HOME = os.getcwd()
INSTR_DIR = os.path.join(HOME, 'instructions')


def main():
    pdf_path = os.path.join(INSTR_DIR, '21_07_3165_i.pdf')
    pdf_url = 'https://www.rceth.by/NDfiles/instr/21_07_3165_i.pdf'
    instr = Instructs(instr_dir=INSTR_DIR,
                      pdf_url=pdf_url, pdf_path=pdf_path)

    print(instr.instr_imgs)


if __name__ == '__main__':
    main()
