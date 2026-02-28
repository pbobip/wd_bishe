"""
MatSAM 零样本推理脚本
基于SAM (Segment Anything Model) 的材料微观结构分割

注意: MatSAM是零样本推理，不需要训练！
它使用SAM的通用分割能力+自动提示生成来分割微观结构
"""
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

# 尝试导入segment_anything
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    HAS_SAM = True
except ImportError:
    HAS_SAM = False
    print("ERROR: segment_anything not installed!")
    print("Please run: pip install segment-anything")
    print("And download SAM weights from: https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")


def detect_crop_height(img):
    """自动检测底部信息栏高度"""
    h, w = img.shape[:2]
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    bottom_h = int(h * 0.75)
    bottom_part = gray[bottom_h:, :]
    edges = cv2.Canny(bottom_part, 50, 150)
    edge_sum = np.sum(edges, axis=1)
    candidates = np.where(edge_sum > w * 0.5 * 255)[0]
    
    if len(candidates) > 0:
        return bottom_h + candidates[0]
    return int(h * 0.85)


def process_single_image(image_path, mask_generator, output_dir):
    """处理单张图像"""
    # 读取图像
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"无法读取: {image_path}")
    
    original_shape = image.shape[:2]
    
    # 裁剪底部信息栏
    crop_h = detect_crop_height(image)
    image_cropped = image[:crop_h, :]
    cropped_shape = image_cropped.shape[:2]
    
    # SAM需要RGB
    image_rgb = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2RGB)
    
    # 生成所有掩码
    masks = mask_generator.generate(image_rgb)
    
    # 合并掩码 - MatSAM策略：
    # 1. 根据面积和形状筛选γ′相 (立方体状，中等面积)
    # 2. 排除太大（背景）和太小（噪声）的区域
    
    combined_mask = np.zeros(cropped_shape, dtype=np.uint8)
    
    total_area = cropped_shape[0] * cropped_shape[1]
    min_area = total_area * 0.0001  # 最小面积
    max_area = total_area * 0.05   # 最大面积 (单个颗粒)
    
    for mask_data in masks:
        mask = mask_data['segmentation']
        area = mask_data['area']
        
        # 面积筛选
        if min_area < area < max_area:
            # 计算紧凑度 (越接近1越像正方形)
            # bbox = mask_data['bbox']  # x, y, w, h
            # compactness = area / (bbox[2] * bbox[3]) if bbox[2] * bbox[3] > 0 else 0
            
            # γ′相通常是浅色区域 - 检查平均亮度
            gray_cropped = cv2.cvtColor(image_cropped, cv2.COLOR_BGR2GRAY)
            mean_intensity = np.mean(gray_cropped[mask])
            
            # γ′相通常比基体更亮
            if mean_intensity > 100:  # 亮度阈值
                combined_mask[mask] = 255
    
    # 创建完整掩码
    full_mask = np.zeros((original_shape[0], original_shape[1]), dtype=np.uint8)
    full_mask[:crop_h, :] = combined_mask
    
    # 读取原始灰度图用于叠加
    original_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    
    # 计算体积分数
    vf = np.sum(combined_mask == 255) / (cropped_shape[0] * cropped_shape[1])
    
    # 保存结果
    stem = Path(image_path).stem
    cv2.imwrite(str(output_dir / f"{stem}_mask.png"), full_mask)
    
    # 叠加可视化
    overlay = cv2.cvtColor(original_gray, cv2.COLOR_GRAY2BGR)
    green_mask = np.zeros_like(overlay)
    green_mask[:, :, 1] = full_mask
    overlay = np.where(green_mask > 0, cv2.addWeighted(overlay, 0.7, green_mask, 0.3, 0), overlay)
    cv2.imwrite(str(output_dir / f"{stem}_overlay.png"), overlay)
    
    return vf


def run_matsam(image_dir, output_dir, sam_checkpoint, model_type="vit_h"):
    """运行MatSAM推理"""
    if not HAS_SAM:
        print("SAM未安装，无法运行MatSAM")
        return
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # 加载SAM模型
    print(f"加载SAM模型: {sam_checkpoint}")
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device)
    
    # 创建自动掩码生成器
    # 针对材料微观结构调整参数
    mask_generator = SamAutomaticMaskGenerator(
        model=sam,
        points_per_side=32,  # 每边采样点数
        pred_iou_thresh=0.88,  # IoU阈值
        stability_score_thresh=0.95,  # 稳定性阈值
        crop_n_layers=1,
        crop_n_points_downscale_factor=2,
        min_mask_region_area=100,  # 最小掩码区域
    )
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_paths = list(Path(image_dir).glob("*.png"))
    total_vf = 0
    success_count = 0
    
    for img_path in tqdm(image_paths, desc="MatSAM Inference"):
        try:
            vf = process_single_image(img_path, mask_generator, output_dir)
            total_vf += vf
            success_count += 1
        except Exception as e:
            print(f"Error {img_path.name}: {e}")
    
    if success_count > 0:
        print(f"Finished. Avg VF: {total_vf/success_count*100:.2f}%")
    else:
        print("No images processed successfully")


if __name__ == "__main__":
    # SAM权重路径 - 需要先下载
    sam_checkpoint = r"C:\Users\pyd111\Desktop\标注3\deep_learning_matsam\sam_vit_h_4b8939.pth"
    
    # 检查权重是否存在
    if not Path(sam_checkpoint).exists():
        print("="*50)
        print("SAM权重文件未找到!")
        print("请下载 ViT-H SAM 模型:")
        print("https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth")
        print(f"并保存到: {sam_checkpoint}")
        print("="*50)
    else:
        run_matsam(
            image_dir=r"C:\Users\pyd111\Desktop\标注3\单晶图像_png",
            output_dir=r"C:\Users\pyd111\Desktop\标注3\deep_learning_matsam\predictions",
            sam_checkpoint=sam_checkpoint,
            model_type="vit_h"
        )
