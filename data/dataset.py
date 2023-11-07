import torch
import numpy as np
import cv2
import os
import random
from torch.utils.data import Dataset

from utils.utils import xywh2xyxy, xyxy2xywh


class VOCDataset(Dataset):
    def __init__(self, label_list, transform=None, is_train=True, net_size=416, net_random=0):
        super(VOCDataset, self).__init__()
        self.is_train = is_train
        self.transform = transform
        self.max_objects = 30

        self.net_size = net_size
        self.net_random = net_random
        self.interps = [
            cv2.INTER_NEAREST,
            cv2.INTER_LINEAR,
            cv2.INTER_AREA,
            cv2.INTER_CUBIC,
            cv2.INTER_LANCZOS4,
        ]

        # for random net_size
        self.batch_count = 0
        self.multiscale_interval = 10
        self.min_scale = 10 * 32
        self.max_scale = 19 * 32

        with open(label_list, 'r') as f:
            image_path_lines = f.readlines()

        self.images_path = []
        self.labels = []
        for image_path_line in image_path_lines:
            image_path = image_path_line.strip().split()[0]
            label_path = image_path.replace('JPEGImages', 'labels').replace('jpg', 'txt')
            if not os.path.exists(label_path):
                continue

            self.images_path.append(image_path)
            with open(label_path, 'r') as f:
                label_lines = f.readlines()

            labels_tmp = np.empty((len(label_lines), 5), dtype=np.float32)
            for i, label_line in enumerate(label_lines):
                labels_tmp[i] = [float(x) for x in label_line.strip().split()]
            self.labels.append(labels_tmp)

        assert len(self.images_path) == len(self.labels), 'images_path\'s length dont match labels\'s length'

    def __getitem__(self, idx):
        image = cv2.cvtColor(cv2.imread(self.images_path[idx]), cv2.COLOR_BGR2RGB)
        labels = self.labels[idx]

        img_h, img_w, _ = image.shape

        if self.is_train and self.transform:
            labels = xywh2xyxy(labels, img_w, img_h)
            image, labels = self.transform(image, labels)
            img_h, img_w, _ = image.shape
            labels = xyxy2xywh(labels, img_w, img_h)

        # resize image
        image = cv2.resize(image, (self.net_size, self.net_size), interpolation=random.choice(self.interps))

        # to torch
        inputs = torch.from_numpy(image.transpose((2, 0, 1))).float().div(255)

        targets = torch.zeros((self.max_objects, 5), dtype=torch.float32)
        if len(labels) < self.max_objects:
            targets[:len(labels), :] = torch.from_numpy(labels)
        else:
            targets = torch.from_numpy(labels)[:self.max_objects, :]

        return inputs, targets

    def collate_fn(self, batch):
        # Selects new image size every tenth batch
        self.batch_count += 1
        if self.net_random and self.batch_count % self.multiscale_interval == 0:
            self.net_size = random.choice(range(self.min_scale, self.max_scale + 1, 32))

        # Get inputs
        inputs, targets = list(zip(*batch))
        inputs = torch.stack(inputs)
        targets = torch.stack(targets)

        return inputs, targets

    def __len__(self):
        return len(self.images_path)