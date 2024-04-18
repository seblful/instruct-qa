import os

from modules.trainer import SegformerTrainer

HOME = os.getcwd()
# Data, dataset dirs
data_dir = os.path.join(HOME, 'data')
dataset_dir = os.path.join(data_dir, 'dataset')


def main():
    segformer_trainer = SegformerTrainer(dataset_dir=dataset_dir,
                                         checkpoint="vikp/surya_layout",
                                         num_epochs=2,
                                         batch_size=1,
                                         num_workers=2)
    segformer_trainer.train()


if __name__ == "__main__":
    main()
