"""
Kaggle SAM ViT-L LoRA 模型预测脚本
"""
import os
import cv2
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm import tqdm
from segment_anything import sam_model_registry
from peft import LoraConfig, get_peft_model

# 配置必须与Kaggle训练时一致
CONFIG = {
    'model_type': 'vit_l',
    'checkpoint_path': r'C:\Users\pyd111\Desktop\标注3\deep_learning_matsam\sam_vit_l_0b3195.pth', # 基础模型
    'lora_path': r'C:\Users\pyd111\Desktop\标注3\deep_learning_sam_lora_kaggle\predictions\sam_lora_best.pth',    # 微调权重
    'image_dir': r'C:\Users\pyd111\Desktop\标注3\单晶图像_png',
    'output_dir': r'C:\Users\pyd111\Desktop\标注3\deep_learning_sam_lora_kaggle\predictions',
    'lora_r': 16,
    'lora_alpha': 32,
    'img_size': 1024,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

class SAMLoRAWrapper(nn.Module):
    def __init__(self, sam_model, lora_r=16, lora_alpha=32):
        super().__init__()
        self.sam = sam_model
        
        # 配置LoRA
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=["qkv"],
            lora_dropout=0.1,
            bias="none",
        )
        self.sam.image_encoder = get_peft_model(self.sam.image_encoder, lora_config)

    def forward(self, images):
        # 仅用于构建结构，推理时直接调研内部组件
        pass

def predict_single(model, img_path, device):
    image = cv2.imread(str(img_path))
    if image is None:
        raise ValueError(f"无法读取: {img_path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_shape = image.shape[:2]
    h, w = original_shape
    
    # 裁剪逻辑
    bottom_h = int(h * 0.75)
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    bottom_part = img_gray[bottom_h:, :]
    edges = cv2.Canny(bottom_part, 50, 150)
    edge_sum = np.sum(edges, axis=1)
    candidates = np.where(edge_sum > w * 0.5 * 255)[0]
    
    crop_h = bottom_h + candidates[0] if len(candidates) > 0 else int(h * 0.85)
    image_cropped = image[:crop_h, :]
    cropped_shape = image_cropped.shape[:2]
    
    # Resize to 1024x1024
    image_resized = cv2.resize(image_cropped, (CONFIG['img_size'], CONFIG['img_size']))
    image_tensor = torch.from_numpy(image_resized).permute(2, 0, 1).float().unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        # 🟢 修复: 添加归一化
        # image_tensor: (1, 3, 1024, 1024) 0-255
        x = model.sam.preprocess(image_tensor)
        image_embeddings = model.sam.image_encoder(x)
        
        sparse_embeddings, dense_embeddings = model.sam.prompt_encoder(
            points=None, boxes=None, masks=None
        )
        
        low_res_masks, _ = model.sam.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=model.sam.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        
        mask_prob = torch.sigmoid(low_res_masks).squeeze().cpu().numpy()
        # 🟢 调优后最佳阈值: 0.3
        mask_binary = (mask_prob > 0.3).astype(np.uint8) * 255
        
        # 恢复尺寸
        mask_resized = cv2.resize(mask_binary, (cropped_shape[1], cropped_shape[0]), interpolation=cv2.INTER_NEAREST)
        
        full_mask = np.zeros((original_shape[0], original_shape[1]), dtype=np.uint8)
        full_mask[:crop_h, :] = mask_resized
        
    return full_mask, cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

def predict_all():
    print(f"Loading Base Model: {CONFIG['model_type']}...")
    if not os.path.exists(CONFIG['checkpoint_path']):
        print(f"正在下载ViT-L权重...")
        os.system(f"wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth -O {CONFIG['checkpoint_path']}")
        
    sam = sam_model_registry[CONFIG['model_type']](checkpoint=CONFIG['checkpoint_path'])
    
    print(f"Applying LoRA (r={CONFIG['lora_r']})...")
    model = SAMLoRAWrapper(sam, lora_r=CONFIG['lora_r'], lora_alpha=CONFIG['lora_alpha'])
    
    print(f"Loading Kaggle Weights from {CONFIG['lora_path']}...")
    checkpoint = torch.load(CONFIG['lora_path'], map_location=CONFIG['device'])
    
    # 加载LoRA权重
    model.sam.image_encoder.load_state_dict(checkpoint['encoder_lora'], strict=False)
    # 加载Decoder权重
    model.sam.mask_decoder.load_state_dict(checkpoint['decoder'])
    
    if 'best_dice' in checkpoint:
        print(f"Model Best Dice: {checkpoint['best_dice']:.4f}")
    
    model.to(CONFIG['device'])
    
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = list(Path(CONFIG['image_dir']).glob("*.png"))
    total_vf = 0
    success_count = 0
    
    for img_path in tqdm(image_paths, desc="Predicting"):
        try:
            mask, original = predict_single(model, img_path, CONFIG['device'])
            
            vf = np.sum(mask == 255) / mask.size 
            total_vf += vf
            
            cv2.imwrite(str(output_dir / f"{img_path.stem}_mask.png"), mask)
            
            overlay = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
            green_mask = np.zeros_like(overlay)
            green_mask[:, :, 1] = mask
            overlay = np.where(green_mask > 0, cv2.addWeighted(overlay, 0.7, green_mask, 0.3, 0), overlay)
            cv2.imwrite(str(output_dir / f"{img_path.stem}_overlay.png"), overlay)
            success_count += 1
        except Exception as e:
            print(f"Error {img_path.name}: {e}")
            import traceback
            traceback.print_exc()
            
    print(f"Finished. Avg VF: {total_vf/success_count*100:.2f}%")

if __name__ == "__main__":
    predict_all()
