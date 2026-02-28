"""
Kaggle SAM ViT-L LoRA 微调脚本 (修复版V2)
================================
变更说明：
1. 切换模型为 **ViT-L** (Large, 308M参数) - 解决ViT-H (Huge)在P100上的显存不足(OOM)问题，同时保持比ViT-B强得多的性能。
2. 保持 **1024x1024** 高分辨率。
3. 如果依然OOM，请在配置中将 `img_size` 改为 768 或 512。

使用说明：
1. 复制本代码到Kaggle Notebook运行。
2. 确保 'kaggle_dataset_fixed.zip' 已上传。
"""

import os
import sys
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

# ----------------- 1. 环境安装 -----------------
print("正在安装依赖...")
os.system("pip install segment-anything peft -q")

from segment_anything import sam_model_registry
from segment_anything.modeling import Sam
from peft import LoraConfig, get_peft_model

# ----------------- 2. 配置 -----------------
CONFIG = {
    # Kaggle数据路径
    'image_dir': '/kaggle/input/i-need-u/单晶图像_png', 
    'label_dir': '/kaggle/input/i-need-u/数据',
    'img_size': 1024, # 尝试保持高分辨率
    'epochs': 100,
    'lr': 1e-4,
    'batch_size': 1,
    'accumulation_steps': 4, # 梯度累积
    'lora_r': 16,
    'lora_alpha': 32,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu")
}

# 路径自动检查
if not os.path.exists(CONFIG['image_dir']):
    if os.path.exists("./images"): # 兼容本地/解压路径
        CONFIG['image_dir'] = "./images"
        CONFIG['label_dir'] = "./labels"
    else:
        print(f"⚠️ 警告: 找不到数据目录 {CONFIG['image_dir']}")

print(f"使用设备: {CONFIG['device']}")

# ----------------- 3. 下载权重 (ViT-L) -----------------
# 切换到ViT-L
WEIGHT_PATH = "sam_vit_l_0b3195.pth"
if not os.path.exists(WEIGHT_PATH):
    print("正在下载SAM ViT-L权重 (1.2GB)...")
    os.system("wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth")
    print("下载完成。")

# ----------------- 4. 数据集定义 -----------------
class GammaPrimeDataset(Dataset):
    def __init__(self, image_dir, label_dir, img_size=1024):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.img_size = img_size
        self.json_files = list(self.label_dir.glob("*.json"))
        print(f"找到 {len(self.json_files)} 个样本")
        
    def _detect_crop_height(self, img):
        h, w = img.shape[:2]
        bottom_h = int(h * 0.75)
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape)==3 else img
        bottom_part = img_gray[bottom_h:, :]
        edges = cv2.Canny(bottom_part, 50, 150)
        edge_sum = np.sum(edges, axis=1)
        candidates = np.where(edge_sum > w * 0.5 * 255)[0]
        if len(candidates) > 0:
            return bottom_h + candidates[0]
        return int(h * 0.85)
    
    def __len__(self):
        return len(self.json_files)
    
    def __getitem__(self, idx):
        json_path = self.json_files[idx]
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        img_stem = json_path.stem
        img_path = list(self.image_dir.glob(f"{img_stem}.*"))[0]
        
        image = cv2.imread(str(img_path))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        crop_h = self._detect_crop_height(image)
        h, w = image.shape[:2]
        image = image[:crop_h, :]
        # 生成Mask
        mask = np.zeros((crop_h, w), dtype=np.uint8)
        for shape in data.get("shapes", []):
            points = np.array(shape["points"], dtype=np.int32)
            points[:, 1] = np.clip(points[:, 1], 0, crop_h - 1)
            cv2.fillPoly(mask, [points], 1)
        
        # 🟢 数据增强 (对9张图的小数据非常重要!)
        if np.random.rand() > 0.5: # 水平翻转
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        if np.random.rand() > 0.5: # 垂直翻转
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        
        k = np.random.randint(0, 4) # 随机旋转 0, 90, 180, 270度
        if k > 0:
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()
        
        # Resize
        image_resized = cv2.resize(image, (self.img_size, self.img_size))
        mask_resized = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float()
        mask_tensor = torch.from_numpy(mask_resized).float().unsqueeze(0)
        
        return image_tensor, mask_tensor

# ----------------- 5. 模型定义 (LoRA) -----------------
class SAMLoRAWrapper(nn.Module):
    def __init__(self, sam_model, lora_r=16, lora_alpha=32):
        super().__init__()
        self.sam = sam_model
        
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["qkv"],
            lora_dropout=0.1,
            bias="none",
        )
        self.sam.image_encoder = get_peft_model(self.sam.image_encoder, lora_config)
        
        for param in self.sam.mask_decoder.parameters():
            param.requires_grad = True
        for param in self.sam.prompt_encoder.parameters():
            param.requires_grad = False
            
        trainable = sum(p.numel() for p in self.sam.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.sam.parameters())
        print(f"可训练参数: {trainable/1e6:.2f}M / {total/1e6:.2f}M ({trainable/total*100:.2f}%)")

    def forward(self, images):
        # 🟢 关键修复: 添加SAM预处理 (归一化 + Pad)
        # images: (B, 3, H, W) 0-255
        x = self.sam.preprocess(images)
        image_embeddings = self.sam.image_encoder(x)
        
        batch_size = images.shape[0]
        
        sparse_embeddings, dense_embeddings = self.sam.prompt_encoder(
            points=None, boxes=None, masks=None
        )
        sparse_embeddings = sparse_embeddings.expand(batch_size, -1, -1)
        dense_embeddings = dense_embeddings.expand(batch_size, -1, -1, -1)
        
        low_res_masks, _ = self.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=self.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return low_res_masks

# ----------------- 6. 训练循环 -----------------
def dice_loss(pred, target, smooth=1e-5):
    pred = torch.sigmoid(pred)
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice = (2 * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()

def train():
    # 切换为ViT-L
    sam = sam_model_registry["vit_l"](checkpoint=WEIGHT_PATH)
    model = SAMLoRAWrapper(sam, lora_r=CONFIG['lora_r'], lora_alpha=CONFIG['lora_alpha'])
    model.to(CONFIG['device'])
    
    dataset = GammaPrimeDataset(CONFIG['image_dir'], CONFIG['label_dir'], img_size=CONFIG['img_size'])
    dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
    
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=CONFIG['lr'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])
    scaler = torch.amp.GradScaler('cuda')
    
    best_dice = 0
    history = {'loss': [], 'dice': []}
    
    print("开始训练...")
    for epoch in range(CONFIG['epochs']):
        model.train()
        epoch_loss = 0
        epoch_dice = 0
        optimizer.zero_grad()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")
        for batch_idx, (images, masks) in enumerate(pbar):
            images = images.to(CONFIG['device'])
            masks = masks.to(CONFIG['device'])
            
            with torch.amp.autocast('cuda'):
                preds = model(images)
                loss = dice_loss(preds, masks) / CONFIG['accumulation_steps']
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % CONFIG['accumulation_steps'] == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            epoch_loss += loss.item() * CONFIG['accumulation_steps']
            
            with torch.no_grad():
                pred_binary = (torch.sigmoid(preds) > 0.5).float()
                dice = (2 * (pred_binary * masks).sum()) / (pred_binary.sum() + masks.sum() + 1e-8)
                epoch_dice += dice.item()
            
            pbar.set_postfix({'loss': loss.item() * CONFIG['accumulation_steps'], 'dice': dice.item()})
        
        if len(dataloader) % CONFIG['accumulation_steps'] != 0:
             scaler.step(optimizer)
             scaler.update()
             optimizer.zero_grad()
        
        scheduler.step()
        avg_dice = epoch_dice / len(dataloader)
        history['dice'].append(avg_dice)
        history['loss'].append(epoch_loss / len(dataloader))
        
        if avg_dice > best_dice:
            best_dice = avg_dice
            # 保存
            model.sam.image_encoder.save_pretrained("sam_lora_encoder")
            torch.save(model.sam.mask_decoder.state_dict(), "sam_decoder.pth")
            torch.save({
                'encoder_lora': model.sam.image_encoder.state_dict(),
                'decoder': model.sam.mask_decoder.state_dict(),
                'best_dice': best_dice
            }, "sam_lora_best.pth")
            print(f"🔥 新最佳Dice: {best_dice:.4f} (已保存)")
            
    plt.figure(figsize=(10, 5))
    plt.plot(history['dice'], label='Dice')
    plt.plot(history['loss'], label='Loss')
    plt.legend()
    plt.title(f"Training History (Best Dice: {best_dice:.4f})")
    plt.savefig("training_curve.png")
    plt.show()
    print("训练结束！请下载 'sam_lora_best.pth' 和 'training_curve.png'")

if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        print(f"出错啦: {e}")
        import traceback
        traceback.print_exc()
