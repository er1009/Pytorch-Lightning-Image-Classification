import albumentations as A
import numpy as np
import pandas as pd
import torch
import torch.utils.data as data
import cv2
import pytorch_lightning as pl


class ClassificationDataset(torch.utils.data.Dataset):

    def __init__(self, csv_path=None, labels_dict=None, transforms=None) -> None:
        super().__init__()
        self.df = pd.read_csv(csv_path)
        self.labels_dict = labels_dict
        self.transform = None if transforms is None else A.Compose(A.from_dict({"transform": transforms}))
        
    def __getitem__(self, idx):
        
        img_path = self.df.iloc[idx]["image_path"]
        label = int(self.labels_dict[self.df.iloc[idx]["label"]])
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255
        
        if self.transform:
            img = self.transform(image=img)["image"]
        
        img = np.moveaxis(img, 2, 0)
        
        return img, label

    def __len__(self):
        return len(self.df)


class ClassificationDataModule(pl.LightningDataModule):
    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams

    def train_dataloader(self):
        data_set = ClassificationDataset(
            **self.hparams.dataset["train"]
        )
        data_loader = data.DataLoader(
            data_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True
        )
        return data_loader

    def val_dataloader(self):
        data_set = ClassificationDataset(
            **self.hparams.dataset["val"]
        )
        data_loader = data.DataLoader(
            data_set,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=True
        )
        return data_loader

