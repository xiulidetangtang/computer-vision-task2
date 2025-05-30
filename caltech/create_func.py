
import os

utils_dataset = '''import os
import random
from torch.utils.data import Dataset
from PIL import Image
import torch

class Caltech101Dataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, max_train_per_class=30):
        self.root_dir = root_dir
        self.transform = transform
        self.max_train_per_class = max_train_per_class
        
        self.classes = []
        if os.path.exists(root_dir):
            for item in os.listdir(root_dir):
                item_path = os.path.join(root_dir, item)
                if os.path.isdir(item_path) and item != 'BACKGROUND_Google':
                    self.classes.append(item)
        
        self.classes.sort()
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        self.samples = self._make_dataset(split)
    
    def _make_dataset(self, split):
        samples = []

        
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.exists(class_dir):
                continue
                
            images = []
            for f in os.listdir(class_dir):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    images.append(f)
            
            random.shuffle(images)
            

            if split == 'train':
                selected_images = images[:min(self.max_train_per_class, len(images))]
            else:  # test
                selected_images = images[self.max_train_per_class:]
            
            for img_name in selected_images:
                img_path = os.path.join(class_dir, img_name)
                if os.path.exists(img_path):
                    samples.append((img_path, self.class_to_idx[class_name]))
        
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")

            image = Image.new('RGB', (224, 224), (0, 0, 0))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
'''

# 2. models/alexnet.py
models_alexnet = '''import torch
import torch.nn as nn
import torchvision.models as models

class AlexNetCaltech101(nn.Module):
    def __init__(self, num_classes=101, pretrained=True):
        super(AlexNetCaltech101, self).__init__()
        
        self.alexnet = models.alexnet(pretrained=pretrained)
        
        self.alexnet.classifier[6] = nn.Linear(4096, num_classes)
        

        if pretrained:

            pass
    
    def forward(self, x):
        return self.alexnet(x)
    
    def freeze_features(self):

        for param in self.alexnet.features.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self):

        for param in self.parameters():
            param.requires_grad = True
'''

# 3. models/resnet18.py
models_resnet18 = '''import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18Caltech101(nn.Module):
    def __init__(self, num_classes=101, pretrained=True):
        super(ResNet18Caltech101, self).__init__()
        

        self.resnet = models.resnet18(pretrained=pretrained)
        

        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
    
    def forward(self, x):
        return self.resnet(x)
    
    def freeze_early_layers(self):

        for param in list(self.resnet.parameters())[:-10]:
            param.requires_grad = False
    
    def unfreeze_all(self):

        for param in self.parameters():
            param.requires_grad = True
'''

# 4. models/train.py
models_train = '''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import time
import copy
from datetime import datetime

def train_model(model, train_loader, val_loader, criterion, optimizer, 
                scheduler=None, num_epochs=50, device='cuda', 
                save_path='checkpoints/', experiment_name='experiment'):
    

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_exp_name = f"{experiment_name}_{timestamp}"
    
    writer = SummaryWriter(f'logs/{full_exp_name}')
    
    print(f"开始训练实验: {full_exp_name}")
    print(f"TensorBoard日志: logs/{full_exp_name}")
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    

    try:
        sample_input = next(iter(train_loader))[0][:1].to(device)
        writer.add_graph(model, sample_input)
    except Exception as e:
        print(f"模型结构记录失败: {e}")
    
    for epoch in range(num_epochs):
        print(f'\\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 60)

        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += inputs.size(0)
            

            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar('Batch/Train_Loss', loss.item(), global_step)
            
            if batch_idx % 20 == 0:
                current_acc = running_corrects.double() / total_samples
                print(f'  Batch {batch_idx:3d}/{len(train_loader)}: Loss={loss.item():.4f}, Acc={current_acc:.4f}')
        
        if scheduler:
            scheduler.step()
            writer.add_scalar('Learning_Rate', scheduler.get_last_lr()[0], epoch)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.item())
        
        model.eval()
        val_running_loss = 0.0
        val_running_corrects = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)
                
                val_running_loss += loss.item() * inputs.size(0)
                val_running_corrects += torch.sum(preds == labels.data)
        
        val_epoch_loss = val_running_loss / len(val_loader.dataset)
        val_epoch_acc = val_running_corrects.double() / len(val_loader.dataset)
        
        val_losses.append(val_epoch_loss)
        val_accs.append(val_epoch_acc.item())
        
        writer.add_scalar('Loss/Train', epoch_loss, epoch)
        writer.add_scalar('Loss/Validation', val_epoch_loss, epoch)
        writer.add_scalar('Accuracy/Train', epoch_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_epoch_acc, epoch)
        
        writer.add_scalars('Loss_Comparison', {
            'Train': epoch_loss,
            'Validation': val_epoch_loss
        }, epoch)
        
        writer.add_scalars('Accuracy_Comparison', {
            'Train': epoch_acc,
            'Validation': val_epoch_acc
        }, epoch)
        
        print(f'Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.4f}')
        print(f'Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.4f}')
        
        if val_epoch_acc > best_acc:
            best_acc = val_epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())
            
            checkpoint_path = f'{save_path}/{full_exp_name}_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'train_losses': train_losses,
                'val_losses': val_losses
            }, checkpoint_path)
            
            print(f'新的最佳模型! 验证准确率: {best_acc:.4f}')
    
    writer.close()
    
    print(f'\\n训练完成!')
    print(f'最佳验证准确率: {best_acc:.4f}')
    
    model.load_state_dict(best_model_wts)
    
    return model, {
        'train_loss': train_losses, 
        'train_acc': train_accs,
        'val_loss': val_losses, 
        'val_acc': val_accs,
        'best_acc': best_acc.item(),
        'experiment_name': full_exp_name
    }
'''

# 5. experiments/finetune.py
experiments_finetune = '''import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.alexnet import AlexNetCaltech101
from models.resnet18 import ResNet18Caltech101
from utils.dataset import Caltech101Dataset
from models.train import train_model

def run_finetune_experiments():

    print("开始Caltech-101微调实验")
    

    dataset_path = 'data/caltech-101/101_ObjectCategories'
    if not os.path.exists(dataset_path):
        print(f"数据集路径不存在: {dataset_path}")
        print("请确保已正确解压Caltech-101数据集")
        return {}
    
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    print("加载数据集...")
    train_dataset = Caltech101Dataset(dataset_path, split='train', transform=train_transform)
    val_dataset = Caltech101Dataset(dataset_path, split='test', transform=val_transform)
    
    print(f"训练集样本数: {len(train_dataset)}")
    print(f"验证集样本数: {len(val_dataset)}")
    print(f"类别数: {len(train_dataset.classes)}")
    
    if len(train_dataset) == 0:
        print("训练集为空，请检查数据集路径和格式")
        return {}
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")
    

    experiments = [
        {
            'name': 'alexnet_pretrained_lr0.001_epochs30',
            'model_type': 'alexnet',
            'pretrained': True,
            'lr': 0.001,
            'epochs': 30,
            'scheduler': 'step'
        },
        {
            'name': 'resnet18_pretrained_lr0.001_epochs30', 
            'model_type': 'resnet18',
            'pretrained': True,
            'lr': 0.001,
            'epochs': 30,
            'scheduler': 'step'
        }
    ]
    
    results = {}
    
    for exp in experiments:
        print(f"\\n{'='*60}")
        print(f"开始实验: {exp['name']}")
        print(f"{'='*60}")
        

        if exp['model_type'] == 'alexnet':
            model = AlexNetCaltech101(num_classes=len(train_dataset.classes), 
                                    pretrained=exp['pretrained'])
        else:
            model = ResNet18Caltech101(num_classes=len(train_dataset.classes), 
                                     pretrained=exp['pretrained'])
        
        model = model.to(device)
        

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=exp['lr'])
        
        if exp['scheduler'] == 'step':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1)
        else:
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=exp['epochs'])
        

        try:
            trained_model, history = train_model(
                model, train_loader, val_loader, criterion, optimizer, 
                scheduler, exp['epochs'], device, 'checkpoints/', exp['name']
            )
            
            results[exp['name']] = history
            print(f"实验 {exp['name']} 完成，最佳准确率: {history['best_acc']:.4f}")
            
        except Exception as e:
            print(f"实验 {exp['name']} 失败: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\\n所有微调实验完成!")
    print(f"查看结果: tensorboard --logdir=logs")
    
    return results
'''

# 6. utils/visualization.py
utils_visualization = '''import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_training_curves(results, save_path='results/'):

    if not results:
        print("没有结果可以绘制")
        return
    
    print(f"绘制训练曲线...")
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    for exp_name, history in results.items():
        epochs = range(1, len(history['train_loss']) + 1)
        

        axes[0, 0].plot(epochs, history['train_loss'], label=f'{exp_name}_train', linewidth=2)
        axes[0, 1].plot(epochs, history['val_loss'], label=f'{exp_name}_val', linewidth=2)
        

        axes[1, 0].plot(epochs, history['train_acc'], label=f'{exp_name}_train', linewidth=2)
        axes[1, 1].plot(epochs, history['val_acc'], label=f'{exp_name}_val', linewidth=2)
    

    axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_title('Validation Loss', fontsize=14, fontweight='bold')
    axes[1, 0].set_title('Training Accuracy', fontsize=14, fontweight='bold')
    axes[1, 1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    
    for ax in axes.flat:
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Epoch')
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/training_curves.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"训练曲线已保存: {save_path}/training_curves.png")

'''

def create_all_files():

    print("创建项目文件结构...")

    dirs = ['utils', 'models', 'experiments', 'results', 'checkpoints', 'logs']
    for dir_name in dirs:
        os.makedirs(dir_name, exist_ok=True)
        init_file = os.path.join(dir_name, '__init__.py')
        if not os.path.exists(init_file):
            with open(init_file, 'w') as f:
                f.write('')

    files = {
        'utils/dataset.py': utils_dataset,
        'models/alexnet.py': models_alexnet,
        'models/resnet18.py': models_resnet18,
        'models/train.py': models_train,
        'experiments/finetune.py': experiments_finetune,
        'utils/visualization.py': utils_visualization
    }
    
    for file_path, content in files.items():
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"创建: {file_path}")
    
    print("所有文件创建完成!")
    print("现在可以运行: python main.py")

if __name__ == "__main__":
    create_all_files()