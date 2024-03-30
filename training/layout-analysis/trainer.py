import os
import torch
import ultralytics


class Trainer():
    def __init__(self,
                 dataset_dir,
                 num_epochs,
                 image_size,
                 batch_size,
                 seed,
                 model_type='n'):

        self.dataset_dir = dataset_dir
        self.num_epochs = num_epochs
        self.image_size = image_size
        self.batch_size = batch_size
        self.seed = seed
        self.model_type = model_type

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.model_yaml = f"yolov8{model_type}-obb.yaml"
        self.model_type = f"yolov8{model_type}-obb.pt"
        self.data_yaml = os.path.join(self.dataset_dir, 'data.yaml')

        self._model = None
        self.is_trained = False

    @property
    def model(self):
        if self._model == None:
            # Build a new model from scratch
            model = ultralytics.YOLO(self.model_yaml)
            # Load a pretrained model
            model = ultralytics.YOLO(self.model_type)

            # # Load from my pretrained model
            # model = ultralytics.YOLO('best3.pt')

            self._model = model

        return self._model

    def train(self):
        '''
        Training model
        '''
        result = self.model.train(data=self.data_yaml,
                                  epochs=self.num_epochs,
                                  imgsz=self.image_size,
                                  batch=self.batch_size,
                                  seed=self.seed,
                                  close_mosaic=0)  # , workers=1 maybe
        self.is_trained = True

        return result

    def validate(self):
        '''
        Validating model on test dataset
        '''

        if self.is_trained == True:
            metrics = self.model.val(
                data=self.data_yaml, imgsz=self.image_size, split='test')

            return metrics

        else:
            return f"Model is not trained yet."
