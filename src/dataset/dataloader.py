import os

import lightning as L
import torch
from torch.utils.data import DataLoader

from dataset.dataset import CocoVideoDataset
from dataset.parser import CocoVidParser
from utils import transforms as T


class CocoVidDataLoader(L.LightningDataModule):
    def __init__(
        self,
        data_dir,
        ann_file,
        img_type,
        start_id=1,
        end_id=186,
        batch_size=1,
        num_workers=0,
        seq_length=1,
        augment=False,
        channel=None,
        resize_ratio=None,
        amp=None,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.img_type = img_type
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.seq_length = seq_length
        self.augment = augment
        self.channel = channel
        self.resize_ratio = resize_ratio

        self.ann_file = os.path.join(data_dir, ann_file)
        self.start_id = start_id
        self.end_id = end_id
        self.coco = CocoVidParser(self.ann_file)

        self.train_videos = self.coco.dataset["info"]["train_videos"]
        self.val_videos = self.coco.dataset["info"]["val_videos"]
        self.test_videos = self.coco.dataset["info"]["test_videos"]
        self.tracking_videos = self.coco.dataset["info"]["tracking_videos"]

        self.mean = self.coco.dataset["info"][f"mean_rgb"]
        self.std = self.coco.dataset["info"][f"std_rgb"]

        if "gray" in img_type:
            if channel is not None:
                self.mean = self.coco.dataset["info"]["mean_rgb"][channel]
                self.std = self.coco.dataset["info"]["std_rgb"][channel]
            else:
                self.mean = self.coco.dataset["info"][f"mean_gray"]
                self.std = self.coco.dataset["info"][f"std_gray"]
        else:
            self.mean = self.coco.dataset["info"][f"mean_rgb"]
            self.std = self.coco.dataset["info"][f"std_rgb"]

        normalize = [
            T.ToTensor(),
            T.Rescale(),
            T.StatefulChannelOscillation(amp=amp) if amp else None,
            T.Normalize(mean=self.mean, std=self.std),
        ]

        normalize = [t for t in normalize if t is not None]

        if "gray" in img_type and self.channel is None:
            normalize.insert(-1, T.Grayscale())

        if resize_ratio:
            normalize.insert(1, T.Resize(resize_ratio))

        if self.augment:
            self.train_transform = (
                normalize[:-1]
                + [
                    T.RandomHorizontalFlip(),
                    T.RandomVerticalFlip(),
                    T.ColorJitter(),
                    T.GaussianBlur((5, 9)),
                ]
                + normalize[-1:]
            )
            self.train_transform = T.Compose(self.train_transform)
        else:
            self.train_transform = T.Compose(normalize)

        self.transform = T.Compose(normalize)

    def setup(self, stage: str):
        if stage == "fit":
            self.train_dataset = CocoVideoDataset(
                coco=self.coco,
                data_dir=self.data_dir,
                start_id=self.start_id,
                end_id=self.end_id,
                videos=self.train_videos,
                channel=self.channel,
                transform=self.train_transform,
            )
            self.val_dataset = CocoVideoDataset(
                coco=self.coco,
                data_dir=self.data_dir,
                start_id=self.start_id,
                end_id=self.end_id,
                videos=self.val_videos,
                channel=self.channel,
                transform=self.transform,
            )
        elif stage == "test" or stage == "predict":
            self.test_dataset = CocoVideoDataset(
                coco=self.coco,
                data_dir=self.data_dir,
                start_id=self.start_id,
                end_id=self.end_id,
                videos=self.test_videos,
                channel=self.channel,
                transform=self.transform,
            )

    def collate_fn(self, data):
        imgs = [minibatch[0] for batch in data for minibatch in batch]
        labels = [minibatch[1] for batch in data for minibatch in batch]

        imgs = torch.stack(imgs, dim=0)

        return imgs, labels

    def train_dataloader(self):
        dataloader = DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return dataloader

    def val_dataloader(self):
        dataloader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return dataloader

    def test_dataloader(self):
        dataloader = DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return dataloader

    def tracking_dataloader(self, dataset):
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
        )

        return dataloader

    def denormalize_image(self, img):
        """Denormalize an image tensor."""

        img = img.squeeze(0)
        mean = torch.tensor(self.mean, dtype=img.dtype, device=img.device).view(
            -1, 1, 1
        )
        std = torch.tensor(self.std, dtype=img.dtype, device=img.device).view(-1, 1, 1)

        output = img * std + mean

        return output.permute(1, 2, 0)
