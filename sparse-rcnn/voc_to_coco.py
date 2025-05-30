import os
import json
import xml.etree.ElementTree as ET
from PIL import Image
import argparse

def voc_to_coco_instance(voc_path, output_path, split='train'):
    # VOC类别映射
    VOC_CLASSES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 
                   'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 
                   'dog', 'horse', 'motorbike', 'person', 'pottedplant', 
                   'sheep', 'sofa', 'train', 'tvmonitor']
    
    coco_data = {
        'info': {'description': 'VOC to COCO format'},
        'licenses': [],
        'images': [],
        'annotations': [],
        'categories': []
    }
    
    # 添加类别信息
    for i, class_name in enumerate(VOC_CLASSES):
        coco_data['categories'].append({
            'id': i + 1,
            'name': class_name,
            'supercategory': 'object'
        })
    
    # 处理图像和标注
    annotation_id = 1
    
    # 读取split文件
    if split == 'train':
        split_files = ['VOC2007/ImageSets/Main/trainval.txt', 
                      'VOC2012/ImageSets/Main/trainval.txt']
    else:
        split_files = ['VOC2007/ImageSets/Main/test.txt']
    
    image_id = 1
    for split_file in split_files:
        with open(os.path.join(voc_path, split_file), 'r') as f:
            image_names = f.read().strip().split('\n')
        
        year = '2007' if 'VOC2007' in split_file else '2012'
        
        for image_name in image_names:
            # 图像信息
            img_path = os.path.join(voc_path, f'VOC{year}/JPEGImages/{image_name}.jpg')
            img = Image.open(img_path)
            width, height = img.size
            
            coco_data['images'].append({
                'id': image_id,
                'file_name': f'{image_name}.jpg',
                'width': width,
                'height': height
            })
            
            # 标注信息
            ann_path = os.path.join(voc_path, f'VOC{year}/Annotations/{image_name}.xml')
            if os.path.exists(ann_path):
                tree = ET.parse(ann_path)
                root = tree.getroot()
                
                for obj in root.findall('object'):
                    class_name = obj.find('name').text
                    if class_name in VOC_CLASSES:
                        bbox = obj.find('bndbox')
                        xmin = int(bbox.find('xmin').text)
                        ymin = int(bbox.find('ymin').text)
                        xmax = int(bbox.find('xmax').text)
                        ymax = int(bbox.find('ymax').text)
                        
                        w = xmax - xmin
                        h = ymax - ymin
                        area = w * h
                        
                        coco_data['annotations'].append({
                            'id': annotation_id,
                            'image_id': image_id,
                            'category_id': VOC_CLASSES.index(class_name) + 1,
                            'bbox': [xmin, ymin, w, h],
                            'area': area,
                            'iscrowd': 0,
                            'segmentation': []  # VOC没有分割mask，留空
                        })
                        annotation_id += 1
            
            image_id += 1
    
    # 保存COCO格式文件
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, f'{split}.json'), 'w') as f:
        json.dump(coco_data, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--voc_path', default='data/VOCdevkit')
    parser.add_argument('--output_path', default='data/voc_ins/annotations')
    args = parser.parse_args()
    
    voc_to_coco_instance(args.voc_path, args.output_path, 'train')
    voc_to_coco_instance(args.voc_path, args.output_path, 'test')
