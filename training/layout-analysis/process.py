import os

from utils import create_train_data

HOME = os.getcwd()
INSTR_DIR = os.path.join(os.path.dirname(
    os.path.dirname(HOME)), 'instructions')
DATA_DIR = os.path.join(HOME, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw-data')


def main():
    create_train_data(instr_dir=INSTR_DIR,
                      save_dir=RAW_DATA_DIR,
                      num_images=30)


if __name__ == "__main__":
    main()
