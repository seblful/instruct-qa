import os

from utils import create_train_data, create_train_data_manually

# Setup paths
HOME = os.getcwd()
INSTR_DIR = os.path.join(HOME, 'instructions')
LS_INPUT_DIR = os.path.join(HOME, 'ls-input-data')


def main():
    # create_train_data(instr_dir=INSTR_DIR,
    #                   save_dir=LS_INPUT_DIR,
    #                   num_images=30)

    create_train_data_manually(instr_dir=INSTR_DIR,
                               save_dir=LS_INPUT_DIR)


if __name__ == "__main__":
    main()
