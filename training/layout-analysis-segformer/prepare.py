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
    # Instantiate DatasetCreator object
    dataset_creator = DatasetCreator(raw_data_dir=raw_data_dir,
                                     images_dir=images_dir,
                                     dataset_dir=dataset_dir,
                                     train_split=0.8)

    # Creating masks from polygons json and split data into train, val and test datasets
    dataset_creator.process()


if __name__ == "__main__":
    main()
