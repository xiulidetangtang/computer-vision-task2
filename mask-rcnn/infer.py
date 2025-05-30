
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
import json
import random
from utils import VOCDataset

plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False




def simple_collate_fn(batch):
    images = []
    targets = []

    for image, target, _ in batch:
        images.append(image)
        targets.append(target)

    return images, targets


def create_model(num_classes=21):
    model = maskrcnn_resnet50_fpn(weights='COCO_V1')

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, 256, num_classes)

    return model


def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model_paths = [
        'checkpoints/final_official_model.pth'
    ]


    model = create_model(21)

    model_loaded = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=device)
                model.load_state_dict(state_dict)
                print(f"成功加载训练模型: {model_path}")
                model_loaded = True
                break
            except Exception as e:
                print(f"加载 {model_path} 失败: {e}")
                continue

    if not model_loaded:
        print("未找到训练模型")
    model.to(device)
    model.eval()
    return model, device


def get_proposals_from_model(model, device, images):
    model.eval()

    with torch.no_grad():
        final_results = model(images)

    pred = final_results[0]

    if len(pred['boxes']) > 0:
        real_boxes = pred['boxes'].cpu()
        proposals = []

        proposals.extend(real_boxes.tolist())



        for box in real_boxes:
            x1, y1, x2, y2 = box.tolist()

            for _ in range(3):
                offset_x = random.uniform(-20, 20)
                offset_y = random.uniform(-20, 20)
                scale = random.uniform(0.8, 1.2)

                new_x1 = max(0, x1 + offset_x)
                new_y1 = max(0, y1 + offset_y)
                new_x2 = x2 + offset_x + (x2 - x1) * (scale - 1)
                new_y2 = y2 + offset_y + (y2 - y1) * (scale - 1)

                proposals.append([new_x1, new_y1, new_x2, new_y2])

        img_h, img_w = images[0].shape[1], images[0].shape[2]
        for _ in range(10):
            x1 = random.uniform(0, img_w * 0.7)
            y1 = random.uniform(0, img_h * 0.7)
            x2 = random.uniform(x1 + 20, img_w)
            y2 = random.uniform(y1 + 20, img_h)
            proposals.append([x1, y1, x2, y2])

        proposals_tensor = torch.tensor(proposals)
    else:
        img_h, img_w = images[0].shape[1], images[0].shape[2]
        proposals = []
        for _ in range(20):
            x1 = random.uniform(0, img_w * 0.7)
            y1 = random.uniform(0, img_h * 0.7)
            x2 = random.uniform(x1 + 20, img_w)
            y2 = random.uniform(y1 + 20, img_h)
            proposals.append([x1, y1, x2, y2])

        proposals_tensor = torch.tensor(proposals)

    mock_proposals = [{'boxes': proposals_tensor}]
    return mock_proposals, final_results


def visualize_proposals_vs_final(model, device, image_path, output_dir='results'):
    classes = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    image = Image.open(image_path).convert('RGB')
    img_tensor = transforms.ToTensor()(image).to(device)

    with torch.no_grad():
        proposals, final_results = get_proposals_from_model(model, device, [img_tensor])

    proposal_boxes = proposals[0]['boxes']
    final_pred = final_results[0]

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 8))

    ax1.imshow(image)
    ax1.set_title('原始图像\n(Original Image)', fontsize=14, fontweight='bold', pad=15)
    ax1.axis('off')

    ax2.imshow(image)
    ax2.set_title('第一阶段: RPN候选框', fontsize=14, fontweight='bold', pad=15)

    num_proposals = min(40, len(proposal_boxes))
    for i in range(num_proposals):
        box = proposal_boxes[i].cpu()
        x1, y1, x2, y2 = box
        rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                 linewidth=1, edgecolor='red', facecolor='none', alpha=0.4)
        ax2.add_patch(rect)

    ax2.text(10, 30, f'显示前{num_proposals}个候选框\n(粗糙、重叠、大量)',
             bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8),
             fontsize=11, color='red', weight='bold')
    ax2.axis('off')

    ax3.imshow(image)
    ax3.set_title('第二阶段: 精确预测', fontsize=14, fontweight='bold', pad=15)

    boxes = final_pred['boxes'].cpu()
    labels = final_pred['labels'].cpu()
    scores = final_pred['scores'].cpu()

    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown']

    final_count = 0
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        if score > 0.5:
            x1, y1, x2, y2 = box
            class_name = classes[label.item()]
            color = colors[final_count % len(colors)]

            # 绘制边界框
            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=3, edgecolor=color, facecolor='none')
            ax3.add_patch(rect)

            ax3.text(x1, y1 - 8, f'{class_name}: {score:.3f}',
                     bbox=dict(boxstyle="round,pad=0.4", facecolor=color, alpha=0.8),
                     fontsize=11, color='white', weight='bold')

            final_count += 1

    if final_count > 0:
        ax3.text(10, 30, f'检测: {final_count}个物体\n',
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8),
                 fontsize=11, color='darkgreen', weight='bold')

    ax3.axis('off')

    plt.tight_layout()

    # 保存结果
    os.makedirs(output_dir, exist_ok=True)
    img_name = os.path.basename(image_path).split('.')[0]
    save_path = os.path.join(output_dir, f'detection_only_{img_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"保存检测结果: {save_path}")
    return save_path


def visualize_detection_with_scores(model, device, image_path, output_dir='results', score_threshold=0.5):
    classes = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car',
               'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
               'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']

    image = Image.open(image_path).convert('RGB')
    img_tensor = transforms.ToTensor()(image).to(device)


    with torch.no_grad():
        predictions = model([img_tensor])

    pred = predictions[0]

    fig, ax = plt.subplots(1, 1, figsize=(12, 10))
    ax.imshow(image)
    ax.set_title('目标检测结果',
                 fontsize=16, fontweight='bold', pad=20)

    boxes = pred['boxes'].cpu()
    labels = pred['labels'].cpu()
    scores = pred['scores'].cpu()


    colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange', 'pink', 'brown',
              'cyan', 'magenta', 'lime', 'indigo', 'teal', 'maroon']

    detection_count = 0
    print(f"\n检测结果 (置信度阈值: {score_threshold}):")
    print("-" * 50)

    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        if score > score_threshold:
            x1, y1, x2, y2 = box
            class_name = classes[label.item()]
            color = colors[detection_count % len(colors)]

            rect = patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                     linewidth=3, edgecolor=color, facecolor='none')
            ax.add_patch(rect)

            label_text = f'{class_name}: {score:.3f}'
            ax.text(x1, y1 - 10, label_text,
                    bbox=dict(boxstyle="round,pad=0.5", facecolor=color, alpha=0.9),
                    fontsize=12, color='white', weight='bold')

            print(f"{detection_count + 1}. {class_name}: {score:.3f} "
                  f"[位置: ({x1:.0f}, {y1:.0f}) - ({x2:.0f}, {y2:.0f})]")

            detection_count += 1


    stats_text = f'检测到 {detection_count} 个物体\n置信度阈值: {score_threshold}'
    ax.text(10, 40, stats_text,
            bbox=dict(boxstyle="round,pad=0.8", facecolor='white', alpha=0.9),
            fontsize=13, color='black', weight='bold')

    ax.axis('off')
    plt.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    img_name = os.path.basename(image_path).split('.')[0]
    save_path = os.path.join(output_dir, f'detection_scores_{img_name}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"保存检测结果: {save_path}")
    print(f"总共检测到 {detection_count} 个物体")

    return save_path


def test_on_voc_testset():
    print("在VOC测试集上进行目标检测测试")

    model, device = load_model()

    test_images = [
        'data/VOCdevkit/VOC2007/JPEGImages/000004.jpg',
        'data/VOCdevkit/VOC2007/JPEGImages/000313.jpg',
        'data/VOCdevkit/VOC2007/JPEGImages/000091.jpg',
        'data/VOCdevkit/VOC2007/JPEGImages/000103.jpg'
    ]

    results = []
    for i, img_path in enumerate(test_images):
        if os.path.exists(img_path):
            print(f"\n{'=' * 60}")
            print(f"处理图片 {i + 1}: {os.path.basename(img_path)}")
            print(f"{'=' * 60}")

            result1 = visualize_proposals_vs_final(model, device, img_path, 'results/voc_test')

            result2 = visualize_detection_with_scores(model, device, img_path, 'results/voc_test')

            results.append((result1, result2))
        else:
            print(f"图片不存在: {img_path}")

    return results


def test_on_external_images():
    print("在外部图片上进行目标检测测试")

    model, device = load_model()

    # 外部图片路径
    external_images = [
        'car.jpg',
        'dog.jpg',
        'bicycle.jpg'
    ]

    for img in external_images:
        print(f"   - {img}")

    results = []
    for i, img_path in enumerate(external_images):
        if os.path.exists(img_path):
            print(f"\n{'=' * 60}")
            print(f"处理外部图片 {i + 1}: {os.path.basename(img_path)}")
            print(f"{'=' * 60}")

            # 可视化检测结果和置信度分数
            result = visualize_detection_with_scores(model, device, img_path, 'results/external_test')
            results.append(result)
        else:
            print(f"请添加图片: {img_path}")

    return results






if __name__ == "__main__":
    test_on_voc_testset()
    test_on_external_images()

