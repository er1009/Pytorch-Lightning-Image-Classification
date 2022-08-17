import os

import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torchmetrics
from pytorch_lightning.callbacks import (LearningRateMonitor, ModelCheckpoint, StochasticWeightAveraging)
from pytorch_lightning.loggers import WandbLogger
from sklearn.metrics import classification_report

from utils import load_yaml, create_args, create_model
from data import ClassificationDataModule
from callbacks import ImagePredictionLogger


class ClassiferFramework(pl.LightningModule):
    """Class to train classificaiton models with Pytorch-Lightning module.
    """
    def __init__(self, hparams) -> None:
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = create_model(**hparams["model_params"])
        self.loss_function = nn.CrossEntropyLoss()
        self.accuracy_fn = torchmetrics.Accuracy(threshold=self.score_threshold, num_classes=hparams["model_params"]["num_classes"], average="macro")

    def forward(self, images):
        return self.model(images)

    def configure_optimizers(self):
        # Support Adam or SGD as optimizers.
        if self.hparams.optimizer == "Adam":
            # AdamW is Adam with a correct implementation of weight decay (see here
            # for details: https://arxiv.org/pdf/1711.05101.pdf)
            optimizer = optim.AdamW(self.parameters(), **self.hparams.optimizer_params)
        elif self.hparams.optimizer == "SGD":
            optimizer = optim.SGD(self.parameters(), **self.hparams.optimizer_params)
        else:
            assert False, f'Unknown optimizer: "{self.hparams.optimizer}"'

        # Reduce the learning rate by 0.1 after 100 and 150 epochs
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, **self.hparams.scheduler)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        images, labels = batch
        preds = self.model(images)
        loss = self.loss_function(preds, labels)
        self.log("train_loss", loss.item(), prog_bar=True, on_step=False, on_epoch=True)
        
        report = classification_report(y_true=labels, y_pred=preds, output_dict=True)
        self.log("train_precision", report["True"]["precision"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_recall", report["True"]["recall"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_f1", report["True"]["f1-score"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("train_accuracy", report["accuracy"], prog_bar=True, on_step=False, on_epoch=True)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        preds = self.model(images).argmax(dim=-1)
        loss = self.loss_function(preds, labels)
        self.log("val_loss", loss.item(), prog_bar=True, on_step=False, on_epoch=True)

        report = classification_report(y_true=labels, y_pred=preds, output_dict=True)
        self.log("val_precision", report["True"]["precision"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_recall", report["True"]["recall"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_f1", report["True"]["f1-score"], prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_accuracy", report["accuracy"], prog_bar=True, on_step=False, on_epoch=True)

def main():
    args = create_args()
    config = load_yaml(args.conf_path)
    os.makedirs(config["wandb"]["save_dir"], exist_ok=True)

    logger = WandbLogger(project=config["hparams"]["experiment_name"],
                         name=config["hparams"]["model_params"]["model"] + config["wandb"]["run_name"],
                         offline=config["wandb"]["dry_run"],
                         save_dir=config["wandb"]["save_dir"])

    # Configure model and data module
    model = ClassiferFramework(config["hparams"])
    data_pl = ClassificationDataModule(config["hparams"])
    
    # Samples required by the custom ImagePredictionLogger callback to log image predictions.
    val_samples = next(iter(data_pl.val_dataloader()))

    # Configure callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(config["wandb"]["save_dir"], "weights"),
        monitor="val_f1",
        mode="max",
        filename="{}".format(config["hparams"]["model_params"]["model"]) + "-"
        + config["hparams"]["optimizer"] + "-{epoch:02d}-{val_image_f1_score:02f}",
        save_top_k=10,
        save_last=True
    )
    images_callback = ImagePredictionLogger(val_samples)

    swa_model = StochasticWeightAveraging(**config["hparams"]["swa"])
    lr_monitor = LearningRateMonitor(logging_interval="step")

    # Create the trainer object
    trainer = pl.Trainer(
        gpus=config["hparams"]["num_gpus"],
        max_epochs=config["hparams"]["epochs"],
        strategy="ddp",
        resume_from_checkpoint=None,
        num_sanity_val_steps=0,
        progress_bar_refresh_rate=2,
        logger=logger,
        callbacks=[checkpoint_callback, swa_model, lr_monitor, images_callback]
    )

    logger.watch(model)
    trainer.fit(model, data_pl)


if __name__ == "__main__":
    main()
