import os
from PIL import Image

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

import pytorch_lightning as pl

from datasets import load_metric

from transformers import SegformerImageProcessor, SegformerConfig

from modules.model import SegformerForRegressionMask


class SegFormerDataset():
    def __init__(self,
                 set_dir,
                 checkpoint="vikp/surya_det2"):
        # Dirs, paths with images, masks and classes
        self.set_dir = set_dir
        self.images_dir = os.path.join(set_dir, 'images')
        self.masks_dir = os.path.join(set_dir, 'masks')
        self.classes_path = os.path.join(set_dir, os.pardir, 'classes.txt')

        # Feauture extractor
        self.processor = SegformerImageProcessor.from_pretrained(checkpoint)

        # Create list if images and labels names
        self.images_listdir = [image for image in os.listdir(
            self.images_dir) if image.endswith('jpg')]
        self.masks_listdir = [label for label in os.listdir(
            self.masks_dir) if label.endswith('.png')]
        # Assert if number of images and masks is the same
        assert len(self.images_listdir) == len(self.masks_listdir)

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

    def __len__(self):
        return len(self.images_listdir)

    def __getitem__(self, idx):
        # Open image and mask
        image = Image.open(os.path.join(
            self.images_dir, self.images_listdir[idx]))
        segmentation_map = Image.open(
            os.path.join(self.masks_dir, self.masks_listdir[idx]))

        # Randomly crop + pad both image and segmentation map to same size
        encoded_inputs = self.processor(
            image, segmentation_map, return_tensors="pt")

        # for k, v in encoded_inputs.items():
        #     encoded_inputs[k].squeeze_()  # remove batch dimension

        return encoded_inputs


class SegformerFinetuner(pl.LightningModule):
    def __init__(self,
                 train_dataset,
                 val_dataset,
                 test_dataset,
                 checkpoint="vikp/surya_det2",
                 batch_size=4,
                 num_workers=2,
                 metrics_interval=100):
        super(SegformerFinetuner, self).__init__()

        # Dataloaders
        self.train_dataloader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        self.val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, num_workers=num_workers)
        self.test_dataloader = DataLoader(
            test_dataset, batch_size=batch_size, num_workers=num_workers)

        # id and label
        self.id2label = train_dataset.id2label
        self.label2id = {v: k for k, v in self.id2label.items()}

        self.num_classes = len(self.id2label.keys())

        self.metrics_interval = metrics_interval

        # Device and model
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model_dtype = [torch.float32,
                            torch.float16][self.device == 'cuda']
        self.model_checkpoint = checkpoint
        self.__model = None

        self.train_mean_iou = load_metric("mean_iou")
        self.val_mean_iou = load_metric("mean_iou")
        self.test_mean_iou = load_metric("mean_iou")

    @property
    def model(self):
        if self.__model is None:
            # Load config and model
            config = SegformerConfig.from_pretrained(self.model_checkpoint)
            model = SegformerForRegressionMask.from_pretrained(
                self.model_checkpoint, torch_dtype=self.model_dtype, config=config)
            # Transfer model to device
            model = model.to(self.device)
            print(
                f"Loading detection model {self.model_checkpoint} on device {self.device} with dtype {self.model_dtype}")

            self.__model = model

        return self.__model

    def forward(self, images, masks):
        outputs = self.model(pixel_values=images, labels=masks)
        return (outputs)

    def training_step(self, batch, batch_nb):

        images, masks = batch['pixel_values'], batch['labels']

        outputs = self(images, masks)

        loss, logits = outputs[0], outputs[1]

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        predicted = upsampled_logits.argmax(dim=1)

        self.train_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(),
            references=masks.detach().cpu().numpy()
        )
        if batch_nb % self.metrics_interval == 0:

            metrics = self.train_mean_iou.compute(
                num_labels=self.num_classes,
                ignore_index=255,
                reduce_labels=False,
            )

            metrics = {
                'loss': loss, "mean_iou": metrics["mean_iou"], "mean_accuracy": metrics["mean_accuracy"]}

            for k, v in metrics.items():
                self.log(k, v)

            return (metrics)
        else:
            return ({'loss': loss})

    def validation_step(self, batch, batch_nb):

        images, masks = batch['pixel_values'], batch['labels']

        outputs = self(images, masks)

        loss, logits = outputs[0], outputs[1]

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        predicted = upsampled_logits.argmax(dim=1)

        self.val_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(),
            references=masks.detach().cpu().numpy()
        )

        return ({'val_loss': loss})

    def validation_epoch_end(self, outputs):
        metrics = self.val_mean_iou.compute(
            num_labels=self.num_classes,
            ignore_index=255,
            reduce_labels=False,
        )

        avg_val_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        val_mean_iou = metrics["mean_iou"]
        val_mean_accuracy = metrics["mean_accuracy"]

        metrics = {"val_loss": avg_val_loss, "val_mean_iou": val_mean_iou,
                   "val_mean_accuracy": val_mean_accuracy}
        for k, v in metrics.items():
            self.log(k, v)

        return metrics

    def test_step(self, batch, batch_nb):

        images, masks = batch['pixel_values'], batch['labels']

        outputs = self(images, masks)

        loss, logits = outputs[0], outputs[1]

        upsampled_logits = nn.functional.interpolate(
            logits,
            size=masks.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        predicted = upsampled_logits.argmax(dim=1)

        self.test_mean_iou.add_batch(
            predictions=predicted.detach().cpu().numpy(),
            references=masks.detach().cpu().numpy()
        )

        return ({'test_loss': loss})

    def test_epoch_end(self, outputs):
        metrics = self.test_mean_iou.compute(
            num_labels=self.num_classes,
            ignore_index=255,
            reduce_labels=False,
        )

        avg_test_loss = torch.stack([x["test_loss"] for x in outputs]).mean()
        test_mean_iou = metrics["mean_iou"]
        test_mean_accuracy = metrics["mean_accuracy"]

        metrics = {"test_loss": avg_test_loss, "test_mean_iou": test_mean_iou,
                   "test_mean_accuracy": test_mean_accuracy}

        for k, v in metrics.items():
            self.log(k, v)

        return metrics

    def configure_optimizers(self):
        return torch.optim.Adam([p for p in self.parameters() if p.requires_grad], lr=2e-05, eps=1e-08)
