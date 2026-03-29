import sys
import os
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from backend.app.services.statistics import statistics_service
from backend.app.schemas.run import TraditionalSegConfig

def test_statistics_engine():
    with open("test_stats_result.txt", "w", encoding="utf-8") as f:
        def prnt(text):
            f.write(str(text) + "\n")

        prnt("========== 开始测试统计内核 ==========")
        
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[0:10, 0:10] = 255
        mask[40:60, 40:60] = 255
        mask[70:80, 70:90] = 255
        
        config = TraditionalSegConfig(
            remove_border=True,
            min_area=10,
            max_area=10000,
            min_solidity=0.0,
            min_circularity=0.0,
            min_roundness=0.0,
            max_aspect_ratio=100.0
        )
        
        prnt("\n[执行解析]...")
        result = statistics_service.summarize(mask, um_per_px=1.0, config=config)
        
        prnt("\n========== 边界剔除与 Lantuéjoul 的权重 ==========")
        prnt(f"识别到的总颗粒数: {result['object_count']}")
        prnt(f"保留下来的有效颗粒数: {result['particle_count']}")
        
        for particle in result['particles']:
            prnt(f" > 有效颗粒 {particle['label']} (中心点 x={particle['centroid_x']:.1f}, y={particle['centroid_y']:.1f}):")
            prnt(f"   包围盒 (w={particle['bbox_w']}, h={particle['bbox_h']}), 面积 = {particle['area_value']}")
            prnt(f"   生存概率推演权重 (Weight) = {particle['weight']:.4f}")
            
        prnt("\n经过加权偏置修复修正后的平均面积:")
        prnt(f" -> 算术简单平均面积: {(400 + 200)/2}") 
        prnt(f" -> 修正后的平均面积: {result['mean_area']:.4f}")
        
        prnt("\n========== 欧氏距离变换通道宽度 ==========")
        cws = result.get('channel_widths', [])
        prnt(f"采样到的有效通道宽度点数量: {len(cws)}")
        if len(cws) > 0:
            prnt(f"通道宽度切片样本 (前10个): {cws[:10]}")
            prnt(f" -> 平均通道宽度: {result['mean_channel_width_x']:.4f}")
        else:
            prnt(" -> (警告) 无法提取任何通道宽度。")

if __name__ == '__main__':
    test_statistics_engine()
