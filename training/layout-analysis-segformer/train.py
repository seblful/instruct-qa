import os

from modules.trainer import SegformerTrainer

HOME = os.getcwd()
# Data, dataset dirs
data_dir = os.path.join(HOME, 'data')
dataset_dir = os.path.join(data_dir, 'dataset')


def main():
    segformer_trainer = SegformerTrainer(dataset_dir=dataset_dir,
                                         image_side=512,
                                         checkpoint="vikp/surya_layout",
                                         num_epochs=10,
                                         batch_size=4,
                                         num_workers=2)
    segformer_trainer.train()


if __name__ == "__main__":
    main()
