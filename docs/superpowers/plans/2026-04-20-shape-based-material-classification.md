# Shape-Based Material Classification Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 基于强化相 `γ′` 的形状描述符，对 100 张镍基单晶 SEM 图像做新的材料学重分类，并输出新的分类文件夹。

**Architecture:** 复用现有裁切后的 SEM 主体图作为输入，先做 `γ′` 二值分割，再提取粒子级几何描述符与图像级方向性描述符，最后用显式材料学规则将图像映射到 5 个组织状态类。输出包含逐图分类表、汇总表、掩膜预览和按类组织的硬链接目录。

**Tech Stack:** Python, OpenCV, NumPy, Pandas, pytest

---

### Task 1: 锁定分类规则接口

**Files:**
- Create: `.codex_tmp/tests/test_shape_based_material_classification.py`
- Create: `.codex_tmp/shape_based_material_classification.py`

- [ ] **Step 1: 先写分类规则单元测试**

```python
def test_assign_material_class_identifies_fine_regular_gamma_prime() -> None:
    features = {
        "largest_component_area_fraction": 0.08,
        "mask_area_fraction": 0.56,
        "component_density": 16.0,
        "orientation_anisotropy": 1.35,
        "mean_equivalent_diameter_um": 0.42,
        "mean_aspect_ratio": 1.18,
        "mean_rectangularity": 0.86,
        "mean_solidity": 0.95,
    }
    label, reason = assign_material_class(features)
    assert label == "A_fine_regular_cuboidal_gamma_prime"
```

- [ ] **Step 2: 运行测试并确认失败**

Run: `python -m pytest .codex_tmp\tests\test_shape_based_material_classification.py -q`
Expected: FAIL，提示缺少 `.codex_tmp/shape_based_material_classification.py`

- [ ] **Step 3: 实现最小分类规则接口**

```python
def assign_material_class(features: dict[str, float]) -> tuple[str, str]:
    ...
```

- [ ] **Step 4: 再次运行测试**

Run: `python -m pytest .codex_tmp\tests\test_shape_based_material_classification.py -q`
Expected: PASS

### Task 2: 实现强化相分割与形状描述符提取

**Files:**
- Modify: `.codex_tmp/shape_based_material_classification.py`

- [ ] **Step 1: 为裁切图实现 `γ′` 分割函数**

```python
def segment_gamma_prime(image: np.ndarray) -> np.ndarray:
    blurred = cv2.GaussianBlur(image, (3, 3), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    return mask
```

- [ ] **Step 2: 实现图像级形状描述符提取**

```python
def extract_shape_features(mask: np.ndarray, um_per_px: float | None) -> dict[str, float | None]:
    return {
        "mask_area_fraction": ...,
        "largest_component_area_fraction": ...,
        "component_density": ...,
        "mean_equivalent_diameter_um": ...,
        "mean_aspect_ratio": ...,
        "mean_rectangularity": ...,
        "mean_solidity": ...,
        "orientation_anisotropy": ...,
    }
```

- [ ] **Step 3: 保存中间掩膜与叠加图**

```python
cv2.imencode(".png", mask)[1].tofile(str(mask_path))
cv2.imencode(".png", overlay)[1].tofile(str(overlay_path))
```

### Task 3: 批量重分类并生成新目录

**Files:**
- Modify: `.codex_tmp/shape_based_material_classification.py`

- [ ] **Step 1: 读取裁切图与尺度表**

```python
cropped_dir = ROOT / "wd_bishe" / "dataset" / "full_png_cropped_xlsx" / "images"
feature_seed = pd.read_csv(ROOT / ".codex_tmp" / "full_png_material_classification" / "image_classification.csv")
```

- [ ] **Step 2: 批量提取特征并分类**

```python
for image_path in cropped_dir.glob("*.png"):
    features = extract_shape_features(mask, um_per_px)
    label, reason = assign_material_class(features)
```

- [ ] **Step 3: 生成结果表和新文件夹**

```python
output_root = ROOT / ".codex_tmp" / "shape_based_material_classification_v3"
by_class_dir = output_root / "by_material_class"
os.link(source_full_png, by_class_dir / class_folder / image_name)
```

- [ ] **Step 4: 运行脚本**

Run: `python .codex_tmp\shape_based_material_classification.py`
Expected: 生成 `csv`、`md`、`by_material_class` 和 `masks/overlays`

### Task 4: 验证结果并抽样复核

**Files:**
- Modify: `.codex_tmp/shape_based_material_classification.py`（若阈值或规则需要微调）

- [ ] **Step 1: 核对单元测试仍为绿色**

Run: `python -m pytest .codex_tmp\tests\test_shape_based_material_classification.py -q`
Expected: PASS

- [ ] **Step 2: 统计新目录的每类数量**

Run: `Get-ChildItem .codex_tmp\shape_based_material_classification_v3\by_material_class | Select Name,@{Name='Count';Expression={(Get-ChildItem $_.FullName -File).Count}}`
Expected: 5 个类别目录，数量总和为 100

- [ ] **Step 3: 读取汇总表并抽样查看代表图**

Run: `Import-Csv .codex_tmp\shape_based_material_classification_v3\summary.csv | Format-Table`
Expected: 每类有明确中文解释与示例图
