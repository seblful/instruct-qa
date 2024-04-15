import os

from trainer import SegFormerDataset

HOME = os.getcwd()
# Data, dataset dirs
data_dir = os.path.join(HOME, 'data')
dataset_dir = os.path.join(data_dir, 'dataset')


def main():
    dataset = SegFormerDataset(set_dir=os.path.join(dataset_dir, 'train'))


if __name__ == "__main__":
    main()
