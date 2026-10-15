import os

import cv2
import numpy as np
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset


class CocoVideoDataset(Dataset):
    def __init__(
        self, coco, data_dir, start_id, end_id, videos, transform, channel=None
    ):
        self.coco = coco
        self.data_dir = data_dir
        self.start_id = start_id
        self.end_id = end_id
        self.videos = videos
        self.transform = transform
        self.channel = channel

        self.data = []

        self.setup()

    def setup(self):
        vid_ids = [
            vid["id"]
            for vid in self.coco.dataset["videos"]
            if vid["name"] in self.videos
        ]

        img_ids = list(self.coco.imgs.keys())

        for img_id in img_ids:
            if self.coco.get_vid_id_from_img_id(img_id) in vid_ids:
                labels = self.load_annotation(img_id)
                self.data.append([img_id, labels])

    def load_annotation(self, img_id):
        annotations = self.coco.imgToAnns[img_id]

        bboxes = []
        category_ids = []
        labels = []

        for annotation in annotations:
            bboxes.append(annotation["bbox"])
            category_ids.append(annotation["category_id"])
            labels.append(annotation["cell_id"])

        bboxes = torch.tensor(bboxes, dtype=torch.float32)
        category_ids = torch.tensor(category_ids)
        labels = torch.tensor(labels)

        annotations = {
            "class_labels": category_ids,
            "boxes": bboxes,
            "labels": labels,
            "img_ids": img_id,
        }

        return annotations

    def load_image(self, img_id):
        img_path = self.coco.loadImgs(img_id)[0]["file_name"]

        image = cv2.imread(os.path.join(self.data_dir, img_path), -1).astype(np.float32)

        if len(image.shape) == 3:
            image = image[:, :, ::-1].copy()

        if self.channel is not None:
            image = image[:, :, self.channel]

        return image

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        output = [self.transform(self.load_image(self.data[idx][0]), self.data[idx][1])]

        return output


class InferenceDataset(Dataset):
    def __init__(self, data_dir, mean, std):
        super().__init__()
        self.data_dir = os.path.join(data_dir, "images")

        normalize = [
            T.ToTensor(),
            T.Normalize(mean=mean, std=std),
        ]

        self.transform = T.Compose(normalize)

        self.images = sorted(
            [
                f
                for f in os.listdir(self.data_dir)
                if f.endswith((".jpg", ".png", ".jpeg", ".tif"))
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = cv2.imread(os.path.join(self.data_dir, self.images[idx]), -1).astype(
            np.float32
        )

        if len(image.shape) == 3:
            image = image[:, :, ::-1].copy()

        image = self.transform(image)
        image = image.unsqueeze(0)

        return image, None
