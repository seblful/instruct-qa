import os
import argparse

from modules.trainer import Trainer

# Create a parser
parser = argparse.ArgumentParser(description="Get some hyperparameters.")

# Get an arg for number of epochs
parser.add_argument("--num_epochs",
                    default=10,
                    type=int,
                    help="Number of training epochs.")

# Get an arg for size of image
parser.add_argument("--image_size",
                    default=640,
                    type=int,
                    help="Size images for training.")

# Get an arg for batch size
parser.add_argument("--batch_size",
                    default=8,
                    type=int,
                    help="Batch size for training.")


# Get an arg for model type
parser.add_argument("--model_type",
                    default='n',
                    type=str,
                    help="Size of YOLO8n model.")

# Get an arg for seed
parser.add_argument("--seed",
                    default=42,
                    type=int,
                    help="Random seed of yolo training")


# Get arguments from the parser
args = parser.parse_args()

# Setup hyperparameters
NUM_EPOCHS = args.num_epochs
IMAGE_SIZE = args.image_size
BATCH_SIZE = args.batch_size
MODEL_TYPE = args.model_type
SEED = args.seed

HOME = os.getcwd()
DATA_DIR = os.path.join(HOME, 'data')
DATASET_DIR = os.path.join(DATA_DIR, 'dataset')


def train():
    "Trains YOLO model"
    trainer = Trainer(dataset_dir=DATASET_DIR,
                      num_epochs=NUM_EPOCHS,
                      image_size=IMAGE_SIZE,
                      batch_size=BATCH_SIZE,
                      seed=SEED,
                      model_type=MODEL_TYPE)

    # Current device
    print(f"Current device is {trainer.device}.")

    # Training
    result = trainer.train()
    metrics = trainer.validate()

    return result, metrics


def main():
    train()


if __name__ == "__main__":
    main()
