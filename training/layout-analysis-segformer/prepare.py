import os

from preparers import DatasetCreator

# Create path vatiables
HOME = os.getcwd()

# Data, dataset dirs
data_dir = os.path.join(HOME, 'data')
raw_data_dir = os.path.join(data_dir, 'raw-data')
dataset_dir = os.path.join(data_dir, 'dataset')

# Directory with images
training_dir = os.path.abspath(os.path.join(HOME, os.pardir))
images_dir = os.path.join(training_dir, 'ls-input-data')


def main():
    pass


if __name__ == "__main__":
    main()
