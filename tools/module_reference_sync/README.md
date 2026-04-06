# 模块源码整理包

这个目录用于收纳两个外部模块资料库里可直接纳入 Git 管理的源码部分。

## 本次整理范围

- 来源 1：`D:\BaiduNetdiskDownload\AI-F`
- 来源 2：`D:\BaiduNetdiskDownload\B站更新（和视频同步更新 日更）\B站更新（和视频同步更新 日更）`

## 整理规则

- 只同步 `.py` 文件
- 保留各来源内部的相对目录结构
- 不上传 `pdf / pptx / mp4 / zip` 等大文件
- 生成完整清单和哈希去重报告，方便后续二次筛选

## 当前结果

### Python 源码

- 总 Python 文件数：`408`
- 总体积：`2,371,584 bytes`
- `AI-F`：`181` 个 `.py`，约 `939,166 bytes`
- `B站更新`：`227` 个 `.py`，约 `1,432,418 bytes`
- 检测到完全重复的哈希组：`26` 组

### PDF 资料

- 原始 PDF 条目总数：`86`
- 去重后准备上传的 PDF 数：以 [pdf_summary.json](/C:/Users/pyd111/Desktop/中期/tmp/module-library-sync/tools/module_reference_sync/pdf_summary.json) 为准
- 按规则分为：
  - `papers`：论文原文、预印本、相关正式 PDF
  - `guides`：说明、教程、创新分析、模块介绍、写作指导等

## 目录说明

- [sources/ai_f_py](/C:/Users/pyd111/Desktop/中期/tmp/module-library-sync/tools/module_reference_sync/sources/ai_f_py)
  - `AI-F` 的源码镜像
- [sources/bilibili_daily_py](/C:/Users/pyd111/Desktop/中期/tmp/module-library-sync/tools/module_reference_sync/sources/bilibili_daily_py)
  - `B站更新` 的源码镜像
- [inventory.csv](/C:/Users/pyd111/Desktop/中期/tmp/module-library-sync/tools/module_reference_sync/inventory.csv)
  - 每个文件的来源、目标路径、大小、SHA256
- [duplicates_by_sha256.csv](/C:/Users/pyd111/Desktop/中期/tmp/module-library-sync/tools/module_reference_sync/duplicates_by_sha256.csv)
  - 完全重复文件报告
- [summary.json](/C:/Users/pyd111/Desktop/中期/tmp/module-library-sync/tools/module_reference_sync/summary.json)
  - 统计汇总
- [sync_module_sources.ps1](/C:/Users/pyd111/Desktop/中期/tmp/module-library-sync/tools/module_reference_sync/sync_module_sources.ps1)
  - 重新生成本目录的脚本
- [pdfs](/C:/Users/pyd111/Desktop/中期/tmp/module-library-sync/tools/module_reference_sync/pdfs)
  - 论文与指导 PDF 的整理目录
- [pdf_inventory.csv](/C:/Users/pyd111/Desktop/中期/tmp/module-library-sync/tools/module_reference_sync/pdf_inventory.csv)
  - PDF 条目清单与 canonical 映射
- [pdf_duplicates_by_sha256.csv](/C:/Users/pyd111/Desktop/中期/tmp/module-library-sync/tools/module_reference_sync/pdf_duplicates_by_sha256.csv)
  - PDF 完全重复报告
- [pdf_summary.json](/C:/Users/pyd111/Desktop/中期/tmp/module-library-sync/tools/module_reference_sync/pdf_summary.json)
  - PDF 汇总统计
- [sync_module_pdfs.ps1](/C:/Users/pyd111/Desktop/中期/tmp/module-library-sync/tools/module_reference_sync/sync_module_pdfs.ps1)
  - 重新生成 PDF 资料目录的脚本

## 说明

这次提交的目标是把外部模块资料库整理成一个适合版本管理的源码与 PDF 资料包，不直接代表这些模块都适合当前实验线。后续如果要继续做模型筛选，建议优先基于 `inventory.csv`、`pdf_inventory.csv` 和重复报告再做一次二次裁剪。
