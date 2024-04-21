import os

from modules.preparers import DatasetCreator
from modules.utils import create_train_data, create_train_data_manually


# Setup hyperparameters
TRAIN_SPLIT = 0.85

# Setup paths
HOME = os.getcwd()
INSTRUCT_QA_DIR = os.path.dirname(os.path.dirname(HOME))
INSTR_DIR = os.path.join(INSTRUCT_QA_DIR, 'data', 'instructions')
LS_INPUT_DIR = os.path.join(INSTRUCT_QA_DIR, 'data', 'ls-input-data')

DATA_DIR = os.path.join(HOME, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw-data')
DATASET_DIR = os.path.join(DATA_DIR, 'dataset')


def main():
    # create_train_data(instr_dir=INSTR_DIR,
    #                   save_dir=LS_INPUT_DIR,
    #                   num_images=30)

    # create_train_data_manually(instr_dir=INSTR_DIR,
    #                            save_dir=LS_INPUT_DIR)

    # Initializing dataset creator and process data(create dataset)
    dataset_creator = DatasetCreator(raw_data_dir=RAW_DATA_DIR,
                                     dataset_dir=DATASET_DIR,
                                     train_split=TRAIN_SPLIT)
    dataset_creator.process()


if __name__ == "__main__":
    main()
