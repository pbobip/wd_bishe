"""
γ′相分割脚本 - 镍基单晶合金SEM图像处理
使用Otsu自动阈值 + 形态学后处理
"""

import cv2
import numpy as np
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time
from scipy import ndimage


def segment_gamma_prime(image_path: str, output_dir: str, info_bar_ratio: float = 0.15, min_area: int = 500) -> dict:
    """
    分割单张SEM图像中的γ′相
    
    Args:
        image_path: 输入图像路径
        output_dir: 输出目录
        info_bar_ratio: 底部信息条占图像高度的比例
        min_area: 最小颗粒面积阈值（像素），小于此值的颗粒将被过滤
    
    Returns:
        dict: 包含分割统计信息
    """
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return {"error": f"无法读取图像: {image_path}"}
    
    original_height = img.shape[0]
    
    # 裁剪底部信息条
    crop_height = int(original_height * (1 - info_bar_ratio))
    img_cropped = img[:crop_height, :]
    
    # 高斯模糊降噪
    img_blur = cv2.GaussianBlur(img_cropped, (3, 3), 0)
    
    # Otsu自动阈值分割
    # γ′相为暗区域，使用THRESH_BINARY_INV使其变为白色
    threshold, mask = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 自适应形态学操作：根据颗粒大小调整核大小
    # 先检测颗粒平均大小
    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask_temp = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    num_labels_temp, _, stats_temp, _ = cv2.connectedComponentsWithStats(mask_temp, connectivity=8)
    
    if num_labels_temp > 1:
        areas = stats_temp[1:, cv2.CC_STAT_AREA]  # 跳过背景
        avg_area = np.median(areas) if len(areas) > 0 else 10000
    else:
        avg_area = 10000
    
    # 根据平均颗粒面积选择闭操作核大小
    # 小颗粒（二次γ′）跳过闭操作避免合并，大颗粒用大核填充空洞
    if avg_area < 2000:  # 二次γ′相 - 细小颗粒，跳过闭操作
        close_size = 0
    elif avg_area < 8000:  # 中等颗粒
        close_size = 5
    else:  # 一次γ′相 - 大颗粒
        close_size = 7
    
    # 开操作：去除小噪点
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_open)
    
    # 闭操作：填充颗粒内部的小空洞（仅对大颗粒）
    if close_size > 0:
        kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close)

    
    # 对每个连通域单独填充内部孔洞，然后过滤小颗粒
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
    
    mask_final = np.zeros_like(mask)
    removed_count = 0
    
    # 对小颗粒图像跳过孔洞填充（避免合并）
    should_fill_holes = avg_area >= 2000
    
    for i in range(1, num_labels):  # 跳过背景（标签0）
        # 提取单个颗粒
        particle_mask = (labels == i).astype(np.uint8) * 255
        
        # 只对大颗粒图像填充孔洞
        if should_fill_holes:
            particle_bool = particle_mask > 0
            particle_filled = ndimage.binary_fill_holes(particle_bool)
            particle_mask = (particle_filled * 255).astype(np.uint8)
        
        # 计算面积
        area = np.sum(particle_mask == 255)
        
        if area >= min_area:
            mask_final = cv2.bitwise_or(mask_final, particle_mask)
        else:
            removed_count += 1
    
    # 使用分水岭算法分离接触的颗粒（仅对大颗粒图像）
    if avg_area >= 2000:
        # 距离变换
        dist_transform = cv2.distanceTransform(mask_final, cv2.DIST_L2, 5)
        
        # 找到局部最大值作为种子点
        dist_max = dist_transform.max()
        if dist_max > 5:  # 只有当距离变换有意义时才分水岭
            # 阈值取距离变换最大值的较低比例，保留更多边缘
            _, sure_fg = cv2.threshold(dist_transform, 0.2 * dist_max, 255, 0)
            sure_fg = np.uint8(sure_fg)
            
            # 确定背景区域 - 使用原始掩码边界
            sure_bg = mask_final.copy()
            
            # 未知区域
            unknown = cv2.subtract(sure_bg, sure_fg)
            
            # 标记连通域
            _, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            
            # 需要3通道图像进行分水岭
            img_color = cv2.cvtColor(img_cropped, cv2.COLOR_GRAY2BGR)
            markers = cv2.watershed(img_color, markers)
            
            # 重建掩码：保留原始边界，只在颗粒内部使用分水岭边界分割
            mask_watershed = mask_final.copy()
            mask_watershed[markers == -1] = 0  # 只移除分水岭边界线
            
            mask_final = mask_watershed

    
    mask = mask_final

    
    # 计算γ′相体积分数
    total_pixels = mask.size
    gamma_prime_pixels = np.sum(mask == 255)
    volume_fraction = gamma_prime_pixels / total_pixels
    
    # 保存掩码
    filename = Path(image_path).stem
    mask_path = os.path.join(output_dir, f"{filename}_mask.png")
    cv2.imwrite(mask_path, mask)
    
    # 创建可视化叠加图（可选）
    img_color = cv2.cvtColor(img_cropped, cv2.COLOR_GRAY2BGR)
    overlay = img_color.copy()
    overlay[mask == 255] = [0, 255, 0]  # γ′相标记为绿色
    vis = cv2.addWeighted(img_color, 0.7, overlay, 0.3, 0)
    
    vis_path = os.path.join(output_dir, f"{filename}_overlay.png")
    cv2.imwrite(vis_path, vis)
    
    return {
        "filename": filename,
        "threshold": threshold,
        "volume_fraction": volume_fraction,
        "removed_particles": removed_count,
        "mask_path": mask_path
    }


def process_single_image(args):
    """处理单张图像的包装函数"""
    image_path, output_dir = args
    return segment_gamma_prime(image_path, output_dir)


def batch_segment(input_dir: str, output_dir: str, num_workers: int = 4):
    """
    批量处理目录中的所有图像
    
    Args:
        input_dir: 输入图像目录
        output_dir: 输出目录
        num_workers: 并行处理线程数
    """
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有PNG图像
    image_files = list(Path(input_dir).glob("*.png"))
    print(f"找到 {len(image_files)} 张图像待处理")
    
    start_time = time.time()
    
    # 并行处理
    args_list = [(str(f), output_dir) for f in image_files]
    
    results = []
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for i, result in enumerate(executor.map(process_single_image, args_list)):
            results.append(result)
            if (i + 1) % 10 == 0:
                print(f"已处理: {i + 1}/{len(image_files)}")
    
    elapsed = time.time() - start_time
    
    # 统计结果
    success_results = [r for r in results if "error" not in r]
    if success_results:
        avg_vf = np.mean([r["volume_fraction"] for r in success_results])
        print(f"\n处理完成!")
        print(f"- 成功: {len(success_results)}/{len(image_files)}")
        print(f"- 平均γ′相体积分数: {avg_vf:.2%}")
        print(f"- 耗时: {elapsed:.1f}秒")
        print(f"- 输出目录: {output_dir}")
    
    return results


if __name__ == "__main__":
    # 配置路径
    INPUT_DIR = r"c:\Users\pyd111\Desktop\标注3\单晶图像_png"
    OUTPUT_DIR = r"c:\Users\pyd111\Desktop\标注3\segmented_masks"
    
    # 执行批量分割
    results = batch_segment(INPUT_DIR, OUTPUT_DIR, num_workers=4)
