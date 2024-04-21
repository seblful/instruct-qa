import os

from modules.trainer import SegformerTrainer

HOME = os.getcwd()
# Data, dataset dirs
data_dir = os.path.join(HOME, 'data')
dataset_dir = os.path.join(data_dir, 'dataset')

# Checkopo
surya_checkpoint = "vikp/surya_layout"
last_checkpoint = "outputs/lightning_logs_csv/version_5/checkpoints/epoch=9-step=700.ckpt"

model_config_path = 'data/config.json'


def main():
    segformer_trainer = SegformerTrainer(dataset_dir=dataset_dir,
                                         image_side=512,
                                         model_checkpoint=surya_checkpoint,
                                         model_config_path=None,
                                         num_epochs=10,
                                         batch_size=4,
                                         num_workers=2)
    segformer_trainer.train()


if __name__ == "__main__":
    main()
