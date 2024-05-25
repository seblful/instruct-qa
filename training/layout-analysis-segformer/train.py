import os
import argparse
import warnings

from modules.trainer import SegformerTrainer

# Ignore warnings
warnings.filterwarnings('ignore')

# Create a parser
parser = argparse.ArgumentParser(description="Get some hyperparameters.")


# Get an arg for number of epochs
parser.add_argument("--num_epochs",
                    default=10,
                    type=int,
                    help="Number of training epochs.")

# Get an arg for size of image
parser.add_argument("--image_side",
                    default=640,
                    type=int,
                    help="Side of image for training.")

# Get an arg for batch size
parser.add_argument("--batch_size",
                    default=8,
                    type=int,
                    help="Batch size for training.")

# Get an arg for num workers
parser.add_argument("--num_workers",
                    default=2,
                    type=int,
                    help="Num workers for data loading.")


# Get arguments from the parser
args = parser.parse_args()

# Setup hyperparameters
NUM_EPOCHS = args.num_epochs
IMAGE_SIDE = args.image_side
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers

HOME = os.getcwd()
# Data, dataset dirs
DATA_DIR = os.path.join(HOME, 'data')
DATASET_DIR = os.path.join(DATA_DIR, 'dataset')

# Checkopoints
surya_checkpoint = "vikp/surya_layout"
last_checkpoint = "outputs/lightning_logs_csv/version_5/checkpoints/epoch=9-step=700.ckpt"

model_config_path = 'data/config.json'


def main():
    # Instantiate model
    segformer_trainer = SegformerTrainer(dataset_dir=DATASET_DIR,
                                         image_side=IMAGE_SIDE,
                                         model_checkpoint=surya_checkpoint,
                                         model_config_path=None,
                                         num_epochs=NUM_EPOCHS,
                                         batch_size=BATCH_SIZE,
                                         num_workers=NUM_WORKERS)
    # Train model
    segformer_trainer.train()

    # Test trained model
    segformer_trainer.test()


if __name__ == "__main__":
    main()
