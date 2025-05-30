


import os
import torch
import torch.nn as nn
import torch.nn.functional as F  # 添加这个导入
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

import torchvision.transforms as transforms
from torchvision.ops import RoIAlign, nms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import xml.etree.ElementTree as ET
import json
from tqdm import tqdm


class VOCDataset(Dataset):

    def __init__(self, data_dir, image_sets, transforms=None):

        self.data_dir = data_dir
        self.transforms = transforms

        self.classes = ['__background__',
                        'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
                        'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
                        'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}


        self.image_ids = []
        for year, image_set in image_sets:
            txt_path = os.path.join(data_dir, year, 'ImageSets', 'Main', f'{image_set}.txt')
            if os.path.exists(txt_path):
                with open(txt_path, 'r') as f:
                    image_ids = f.read().strip().split('\n')
                    for img_id in image_ids:
                        if img_id.strip():
                            self.image_ids.append((year, img_id.strip()))
            else:
                print(f"警告: 文件不存在 {txt_path}")

        print(f"加载了 {len(self.image_ids)} 张图片")

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        year, img_id = self.image_ids[idx]

        img_path = os.path.join(self.data_dir, year, 'JPEGImages', f'{img_id}.jpg')

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"加载图像失败: {img_path}, 错误: {e}")
            image = Image.new('RGB', (800, 800), (128, 128, 128))

        ann_path = os.path.join(self.data_dir, year, 'Annotations', f'{img_id}.xml')
        target = self.parse_annotation(ann_path, image.size, idx)

        if self.transforms:
            image = self.transforms(image)

        return image, target, img_id

    def parse_annotation(self, ann_path, img_size, idx):
        img_width, img_height = img_size

        boxes = []
        labels = []

        if os.path.exists(ann_path):
            try:
                tree = ET.parse(ann_path)
                root = tree.getroot()

                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    if class_name not in self.class_to_idx:
                        continue

                    difficult = obj.find('difficult')
                    if difficult is not None and int(difficult.text) == 1:
                        continue

                    label = self.class_to_idx[class_name]

                    bbox = obj.find('bndbox')
                    xmin = max(0, float(bbox.find('xmin').text))
                    ymin = max(0, float(bbox.find('ymin').text))
                    xmax = min(img_width, float(bbox.find('xmax').text))
                    ymax = min(img_height, float(bbox.find('ymax').text))

                    if xmax > xmin and ymax > ymin:
                        boxes.append([xmin, ymin, xmax, ymax])
                        labels.append(label)

            except Exception as e:
                print(f"解析标注文件失败: {ann_path}, 错误: {e}")

        if len(boxes) == 0:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)
            masks = torch.zeros((0, 28, 28), dtype=torch.float32)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.long)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.long)

            masks = []
            for box in boxes:
                mask = torch.zeros((28, 28), dtype=torch.float32)

                mask[4:24, 4:24] = 1.0
                masks.append(mask)
            masks = torch.stack(masks)

            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
            iscrowd = torch.zeros((len(boxes),), dtype=torch.long)

        target = {
            'boxes': boxes,
            'labels': labels,
            'masks': masks,
            'image_id': torch.tensor([idx], dtype=torch.long),
            'area': area,
            'iscrowd': iscrowd
        }

        return target

