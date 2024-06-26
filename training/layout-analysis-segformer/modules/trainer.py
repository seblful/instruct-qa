import os
import time
import json
from PIL import Image

import numpy as np

from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

import evaluate

from transformers import SegformerImageProcessor, SegformerConfig

from modules.model import SegformerForRegressionMask


class SegFormerDataset(Dataset):
    def __init__(self,
                 set_dir,
                 image_side,
                 checkpoint="vikp/surya_layout"):
        # Dirs, paths with images, masks and classes
        self.set_dir = set_dir
        self.images_dir = os.path.join(set_dir, 'images')
        self.masks_dir = os.path.join(set_dir, 'masks')

        # Feauture extractor
        self.processor = SegformerImageProcessor.from_pretrained(
            checkpoint, do_reduce_labels=False)
        self.processor.size = {"height": image_side, "width": image_side}

        # Create list if images and labels names
        self.images_listdir = [image for image in os.listdir(
            self.images_dir) if image.endswith('jpg')]
        self.masks_listdir = [label for label in os.listdir(
            self.masks_dir) if label.endswith('.png')]
        # Assert if number of images and masks is the same
        assert len(self.images_listdir) == len(self.masks_listdir)

    def __len__(self):
        return len(self.images_listdir)

    def __getitem__(self, idx):
        # Open image
        image = Image.open(os.path.join(
            self.images_dir, self.images_listdir[idx]))

        # Open mask
        mask = Image.open(
            os.path.join(self.masks_dir, self.masks_listdir[idx]))

        # Randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.processor(
            image, mask, return_tensors="pt")

        for k, v in encoded_inputs.items():
            encoded_inputs[k].squeeze_()  # remove batch dimension

        return encoded_inputs


class SegmentationDataModule(pl.LightningDataModule):
    def __init__(self,
                 dataset_dir,
                 image_side,
                 batch_size,
                 num_workers):
        super().__init__()
        self.dataset_dir = dataset_dir
        self.classes_path = os.path.join(dataset_dir, 'classes.txt')

        self.image_side = image_side

        self.batch_size = batch_size
        self.num_workers = num_workers

        # id and label
        self.__id2label = None
        self.__label2id = None

    @property
    def id2label(self):
        if self.__id2label is None:
            # Open classes_path and extract names from there
            with open(self.classes_path, 'r') as classes_file:
                classes = [i.split('\n')[0] for i in classes_file.readlines()]
                id2label = {k: v for k, v in enumerate(classes)}

            self.__id2label = id2label

        return self.__id2label

    @property
    def label2id(self):
        if self.__label2id is None:
            self.__label2id = {v: k for k, v in self.id2label.items()}

        return self.__label2id

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = SegFormerDataset(
                set_dir=os.path.join(self.dataset_dir, 'train'),
                image_side=self.image_side)
            self.val_dataset = SegFormerDataset(
                set_dir=os.path.join(self.dataset_dir, 'val'),
                image_side=self.image_side)

        if stage == 'test' or stage is None:
            self.test_dataset = SegFormerDataset(
                set_dir=os.path.join(self.dataset_dir, 'test'),
                image_side=self.image_side)

    def train_dataloader(self):
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          shuffle=True,
                          num_workers=self.num_workers,
                          persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          persistent_workers=True)


class SegformerFinetuner(pl.LightningModule):
    def __init__(self,
                 model_checkpoint,
                 model_config_path,
                 id2label,
                 image_side):
        super(SegformerFinetuner, self).__init__()

        # id and label
        self.id2label = id2label
        self.label2id = {v: k for k, v in self.id2label.items()}
        self.num_labels = len(id2label.keys())
        self.image_side = image_side

        # Device and model
        self.model_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_dtype = torch.float16 if self.model_device == 'cuda' else torch.float32

        self.model_checkpoint = model_checkpoint
        self.model_config_path = model_config_path if model_config_path is not None else "vikp/surya_layout"

        self.model = self.load_model()

        # Metrics
        self.train_mean_iou = evaluate.load("mean_iou")
        self.val_mean_iou = evaluate.load("mean_iou")
        self.test_mean_iou = evaluate.load("mean_iou")

    def load_model(self):
        # Config
        config = SegformerConfig.from_pretrained(self.model_config_path)

        config.id2label = self.id2label
        config.label2id = self.label2id
        config.torch_dtype = self.model_dtype
        config.num_labels = self.num_labels
        config.image_size = self.image_side

        # Save config
        config.save_pretrained('./')

        # Model
        if self.model_checkpoint.endswith(".ckpt"):
            # Load model and checkpoint
            model = SegformerForRegressionMask(config=config)
            checkpoint = torch.load(self.model_checkpoint)

            # Create a new state dictionary without the "model." prefix
            new_state_dict = {}

            for key, value in checkpoint['state_dict'].items():
                new_key = key.removeprefix("model.")
                new_state_dict[new_key] = value

            # Load state dict
            model.load_state_dict(new_state_dict)

        else:
            model = SegformerForRegressionMask.from_pretrained(self.model_checkpoint,
                                                               config=config,
                                                               ignore_mismatched_sizes=True)

        model = model.to(self.model_device)

        return model

    def forward(self, pixel_values, labels):
        outputs = self.model(pixel_values=pixel_values, labels=labels)
        return (outputs)

    def on_train_start(self):
        self.start_time = time.time()

    def on_train_end(self):
        total_time = time.time() - self.start_time
        metrics = {'final_epoch': self.current_epoch,
                   'training_time': total_time}
        with open('segformer_hyperparameters.json', 'w') as f:
            json.dump(metrics, f)

    def training_step(self, batch, batch_idx):
        images, masks = batch['pixel_values'], batch['labels']
        outputs = self(images, masks)
        loss, logits = outputs.loss, outputs.logits
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)
        metrics = self.train_mean_iou._compute(
            predictions=predicted.detach().cpu().numpy(),
            references=masks.detach().cpu().numpy(),
            num_labels=self.num_labels,
            ignore_index=254,
            reduce_labels=False,
        )
        # Extract per category metrics and convert to list if necessary (pop before defining the metrics dictionary)
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()

        # Re-define metrics dict to include per-category metrics directly
        metrics = {
            'loss': loss,
            "mean_iou": metrics["mean_iou"],
            "mean_accuracy": metrics["mean_accuracy"],
            **{f"accuracy_{self.id2label[i]}": v for i, v in enumerate(per_category_accuracy)},
            **{f"iou_{self.id2label[i]}": v for i, v in enumerate(per_category_iou)}
        }
        for k, v in metrics.items():
            self.log(k, v, sync_dist=True, on_epoch=True, logger=True)
        return (metrics)

    def validation_step(self, batch, batch_idx):
        images, masks = batch['pixel_values'], batch['labels']
        outputs = self(images, masks)
        loss, logits = outputs.loss, outputs.logits
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=masks[0].shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)
        metrics = self.val_mean_iou._compute(
            predictions=predicted.detach().cpu().numpy(),
            references=masks.detach().cpu().numpy(),
            num_labels=self.num_labels,
            ignore_index=254,
            reduce_labels=False,
        )
        # Extract per category metrics and convert to list if necessary (pop before defining the metrics dictionary)
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()

        # Re-define metrics dict to include per-category metrics directly
        metrics = {
            'loss': loss,
            "mean_iou": metrics["mean_iou"],
            "mean_accuracy": metrics["mean_accuracy"],
            **{f"accuracy_{self.id2label[i]}": v for i, v in enumerate(per_category_accuracy)},
            **{f"iou_{self.id2label[i]}": v for i, v in enumerate(per_category_iou)}
        }
        for k, v in metrics.items():
            self.log(k, v, sync_dist=True)
        return (metrics)

    def test_step(self, batch, batch_idx):
        images, masks = batch['pixel_values'], batch['labels']
        outputs = self(images, masks)
        loss, logits = outputs.loss, outputs.logits
        upsampled_logits = nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )
        predicted = upsampled_logits.argmax(dim=1)
        metrics = self.test_mean_iou._compute(
            predictions=predicted.detach().cpu().numpy(),
            references=masks.detach().cpu().numpy(),
            num_labels=self.num_labels,
            ignore_index=254,
            reduce_labels=False,
        )
        # Extract per category metrics and convert to list if necessary (pop before defining the metrics dictionary)
        per_category_accuracy = metrics.pop("per_category_accuracy").tolist()
        per_category_iou = metrics.pop("per_category_iou").tolist()

        # Re-define metrics dict to include per-category metrics directly
        metrics = {
            'loss': loss,
            "mean_iou": metrics["mean_iou"],
            "mean_accuracy": metrics["mean_accuracy"],
            **{f"accuracy_{self.id2label[i]}": v for i, v in enumerate(per_category_accuracy)},
            **{f"iou_{self.id2label[i]}": v for i, v in enumerate(per_category_iou)}
        }
        for k, v in metrics.items():
            self.log(k, v, sync_dist=True)
        return (metrics)

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=0.001)


class SegformerTrainer():
    def __init__(self,
                 dataset_dir,
                 image_side=1024,
                 model_checkpoint="vikp/surya_layout",
                 model_config_path=None,
                 num_epochs=10,
                 batch_size=4,
                 num_workers=2):

        # Data module
        self.data_module = SegmentationDataModule(dataset_dir=dataset_dir,
                                                  image_side=image_side,
                                                  batch_size=batch_size,
                                                  num_workers=num_workers)

        # Finetuner
        self.segformer_finetuner = SegformerFinetuner(model_checkpoint=model_checkpoint,
                                                      model_config_path=model_config_path,
                                                      id2label=self.data_module.id2label,
                                                      image_side=image_side)

        # Callbacks
        self.early_stop_callback = EarlyStopping(monitor="loss",
                                                 min_delta=0.00,
                                                 patience=10,
                                                 verbose=True,
                                                 mode="min")

        self.checkpoint_callback = ModelCheckpoint(save_top_k=1,
                                                   monitor="loss",
                                                   every_n_epochs=1,  # Save the model at every epoch
                                                   save_on_train_epoch_end=True)  # Ensure saving happens at the end of a training epoch

        # Loggers
        self.logger = TensorBoardLogger(
            "tb_logger", name="segformer_lightning_v2")
        self.logger_csv = pl.loggers.CSVLogger(
            "outputs", name="lightning_logs_csv")

        # Set precision
        torch.set_float32_matmul_precision("medium")

        # Trainer
        self.trainer = pl.Trainer(logger=self.logger_csv,
                                  strategy="auto",
                                  accelerator='gpu',
                                  precision="16-mixed",
                                  callbacks=[self.early_stop_callback,
                                             self.checkpoint_callback],
                                  max_epochs=num_epochs)

    def train(self):
        self.trainer.fit(self.segformer_finetuner, self.data_module)

    def test(self):
        self.trainer.test(self.segformer_finetuner, self.data_module)
