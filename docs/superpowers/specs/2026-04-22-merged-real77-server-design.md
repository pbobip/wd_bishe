# Merged Real77 Server Design

**目标**

把三份真实标签数据整合为一个统一监督训练集，在服务器完成训练后，固定对本地那 `100` 张已裁剪图执行最终推理，并把叠加图与统计结果回传本地。

**数据源**

- `analysis_same_teacher_nocap_095\1`：`26` 张 `LabelMe json`
- `全部已完成标注图像_精修底图`：`16` 张 `LabelMe json`
- `prepared_nasa_super_secondary_train`：`35` 张已准备好的 NASA 二分类训练样本

**设计选择**

1. 前两份 `LabelMe` 数据先用现有 `preparation.py` 转为标准 `prepared_root`
2. 三份 `prepared_root` 再用新增 `prepare_merged_supervised.py` 合并成统一训练集
3. 合并时给每个来源加 `source_alias` 前缀，消除 stem 冲突
4. 交叉验证按来源分组做平衡随机拆分，避免某一折完全缺少某个来源
5. 服务器不上传整个仓库，只上传最小运行 bundle：
   - `experiments/`
   - `backend/app`
   - 三份训练数据
   - 最终推理输入的 `100` 张裁剪图
   - `MicroNet` 权重
6. 训练结束后，自动选择 `crossval_summary.json` 里验证指标最好的 fold 做最终推理

**最终交付**

- 远端输出目录：`experiments/mbu_netpp/outputs/merged_real77_supervised/final_infer_100`
- 远端交付目录：`results/server_delivery/merged_real77_final_infer_100`
- 本地回传内容：
  - `masks/`
  - `overlays/`
  - `stats/`
  - `summary.csv`
  - `summary.xlsx`
