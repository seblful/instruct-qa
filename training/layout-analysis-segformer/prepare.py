import os

from modules.preparers import DatasetCreator

# Create path vatiables
HOME = os.getcwd()

# Data, dataset dirs
DATA_DIR = os.path.join(HOME, 'data')
RAW_DATA_DIR = os.path.join(DATA_DIR, 'raw-data')
DATASET_DIR = os.path.join(DATA_DIR, 'dataset')

# Directory with images
INSTRUCT_QA_DIR = os.path.dirname(os.path.dirname(HOME))
LS_INPUT_DIR = os.path.join(INSTRUCT_QA_DIR, 'data', 'ls-input-data')


def main():
    # Instantiate DatasetCreator object
    dataset_creator = DatasetCreator(raw_data_dir=RAW_DATA_DIR,
                                     images_dir=LS_INPUT_DIR,
                                     dataset_dir=DATASET_DIR,
                                     train_split=0.8)

    # Creating masks from polygons json and split data into train, val and test datasets
    dataset_creator.process()


if __name__ == "__main__":
    main()
