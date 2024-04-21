import os
import warnings

from modules.trainer import SegformerTrainer

# Ignore warnings
warnings.filterwarnings('ignore')

HOME = os.getcwd()
# Data, dataset dirs
data_dir = os.path.join(HOME, 'data')
dataset_dir = os.path.join(data_dir, 'dataset')

# Checkopoints
surya_checkpoint = "vikp/surya_layout"
last_checkpoint = "outputs/lightning_logs_csv/version_5/checkpoints/epoch=9-step=700.ckpt"

model_config_path = 'data/config.json'


def main():
    # Instantiate model
    segformer_trainer = SegformerTrainer(dataset_dir=dataset_dir,
                                         image_side=512,
                                         model_checkpoint=surya_checkpoint,
                                         model_config_path=None,
                                         num_epochs=10,
                                         batch_size=4,
                                         num_workers=2)
    # Train model
    segformer_trainer.train()

    # Test trained model
    segformer_trainer.test()


if __name__ == "__main__":
    main()
