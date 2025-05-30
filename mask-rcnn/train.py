
import torch
import torchvision
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from tqdm import tqdm
import numpy as np
import os
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import json
import random
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']  # 支持中文字体
plt.rcParams['axes.unicode_minus'] = False

from utils import VOCDataset


class MaskRCNNWithProposals(torch.nn.Module):

    def __init__(self, num_classes=21):
        super().__init__()
        self.backbone_model = maskrcnn_resnet50_fpn(weights='COCO_V1')

        in_features = self.backbone_model.roi_heads.box_predictor.cls_score.in_features
        self.backbone_model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        in_features_mask = self.backbone_model.roi_heads.mask_predictor.conv5_mask.in_channels
        self.backbone_model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    def forward(self, images, targets=None):
        if self.training:
            return self.backbone_model(images, targets)
        else:

            return self.get_proposals_and_results(images)

    def get_proposals_and_results(self, images):

        self.backbone_model.eval()

        features = self.backbone_model.backbone(images)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])


        proposals, _ = self.backbone_model.rpn(images, features)

        final_results = self.backbone_model(images)

        return {
            'proposals': proposals,
            'final_results': final_results
        }


def simple_collate_fn(batch):
    images = []
    targets = []

    for image, target, _ in batch:
        images.append(image)
        targets.append(target)

    return images, targets


def create_model(num_classes=21):
    return MaskRCNNWithProposals(num_classes)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    def get_transform():
        return transforms.Compose([transforms.ToTensor()])

    train_dataset = VOCDataset(
        data_dir='data/VOCdevkit',
        image_sets=[('VOC2007', 'trainval')],
        transforms=get_transform()
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=2,
        shuffle=True,
        num_workers=0,
        collate_fn=simple_collate_fn
    )

    model = create_model(21)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    os.makedirs('checkpoints', exist_ok=True)

    for epoch in range(6):
        print(f"\nEpoch {epoch + 1}/6")

        model.train()
        total_loss = 0

        pbar = tqdm(train_loader, desc=f'Training')

        for batch_idx, (images, targets) in enumerate(pbar):
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v
                        for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

            total_loss += losses.item()

            if batch_idx % 100 == 0:
                pbar.set_postfix({'loss': f'{losses.item():.4f}'})

        print(f"平均损失: {total_loss / len(train_loader):.4f}")
        lr_scheduler.step()


        if (epoch + 1) % 2 == 0:
            torch.save(model.state_dict(), f'checkpoints/extended_maskrcnn_epoch_{epoch + 1}.pth')

    torch.save(model.state_dict(), 'checkpoints/final_extended_maskrcnn.pth')
    print("训练完成!")
    return model







if __name__ == "__main__":
    train()
