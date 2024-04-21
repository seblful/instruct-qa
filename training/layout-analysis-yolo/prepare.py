import os

from modules.preparers import DatasetCreator


# Setup hyperparameters
TRAIN_SPLIT = 0.85

# Setup paths
HOME = os.getcwd()
DATA_DIR = os.path.join(HOME, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw-data')
DATASET_DIR = os.path.join(DATA_DIR, 'dataset')


def main():

    # Initializing dataset creator and process data(create dataset)
    dataset_creator = DatasetCreator(raw_data_dir=RAW_DATA_DIR,
                                     dataset_dir=DATASET_DIR,
                                     train_split=TRAIN_SPLIT)
    dataset_creator.process()


if __name__ == "__main__":
    main()
