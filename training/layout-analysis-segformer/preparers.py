import os


class DatasetCreator():
    def __init__(self,
                 raw_data_dir,
                 images_dir,
                 dataset_dir,
                 train_split=0.8) -> None:
        self.inputs_dir = raw_data_dir
        self.images_dir = images_dir
        self.dataset_dir = dataset_dir

        self.json_polygon_path = os.path.join(
            raw_data_dir, 'polygon_labels.json')
        self.classes_path = os.path.join(raw_data_dir, 'classes.txt')

        self.train_split = train_split

        self.__label2id = None
