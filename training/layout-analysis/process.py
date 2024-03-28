import os

from utils import create_train_data, create_train_data_manually

HOME = os.getcwd()
INSTR_DIR = os.path.join(os.path.dirname(
    os.path.dirname(HOME)), 'instructions')
DATA_DIR = os.path.join(HOME, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'ls-input-data')


def main():
    # create_train_data(instr_dir=INSTR_DIR,
    #                   save_dir=RAW_DATA_DIR,
    #                   num_images=30)

    create_train_data_manually(instr_dir=INSTR_DIR,
                               save_dir=RAW_DATA_DIR)


if __name__ == "__main__":
    main()
