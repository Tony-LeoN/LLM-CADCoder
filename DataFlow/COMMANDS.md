# DataFlow Processing Commands

本文档汇总 LLM-CADCoder 当前数据流的处理命令，用于在服务器或本地项目根目录下按阶段生成产物。

默认运行位置：

```bash
cd ~/PycharmProjects/LLMCAD-coder
```

本项目 Python 包位于 `src/` 下。若未安装为 editable package，先设置：

```bash
export PYTHONPATH=src
```

Windows PowerShell 本地调试可用：

```powershell
$env:PYTHONPATH="src"
```

注意：当前 VLM、CadQuery、CUDA 相关命令建议在服务器环境中运行；本地主要用于维护代码、文档和数据流结构。

## 0. 阶段总览

```text
01.RawPDFWithSTEP
-> 02.RawPNG
-> 03.LayoutAnalysis
-> 04.CleanPNG
-> 05.ViewDetection
-> 06.SingleViews
-> 07.ViewClassification
-> 08.Multi-viewFeatureExtraction
-> 09.Cross-viewGeometricReasoning
-> 10.StructuredCADRepresentation
-> 11.CADProgram
```

当前命令状态：

| 阶段转换 | 状态 | 说明 |
| --- | --- | --- |
| `01 -> sample index` | 已实现 | 扫描 PDF/STEP 配对，生成 `data/samples.csv` |
| `01 -> 02` | 已实现 | 单个或批量 PDF 渲染为 PNG |
| `02 -> 03/04` | 已实现 | 页面级 layout 分析与 clean PNG |
| `04 -> 05` | 待接入 | 计划接入 SketchSegment `ViewBlockDetector` |
| `05 -> 06` | 待接入 | 根据视图 bbox 裁剪单视图 |
| `06 -> benchmark` | 已实现 | 使用 single-view crops 跑 VLM 小模型任务 |
| `06 + experiments -> 10/11 prompt` | 已实现原型 | 外部 crops 原型闭环 |
| `10 -> 11 draft` | 已实现原型 | 规则化 CadQuery 草稿 |
| `10/11 prompt -> 11 LLM code` | 已实现原型 | VLM/LLM 直接生成 CadQuery 代码 |

## 1. 生成样本索引

用途：扫描 `01.RawPDFWithSTEP` 下的 PDF 与 STEP/STP 文件，生成样本清单。

输入：

```text
DataFlow/01.RawPDFWithSTEP/
```

命令：

```bash
python -m vlm_cadcoder.cli build-sample-index \
  --raw-dir DataFlow/01.RawPDFWithSTEP \
  --output data/samples.csv
```

输出：

```text
data/samples.csv
```

说明：该命令不生成 DataFlow 阶段目录，只用于建立样本清单。

## 2. `01.RawPDFWithSTEP -> 02.RawPNG`

用途：读取 `01.RawPDFWithSTEP` 中的 PDF，生成对应 PNG 到 `02.RawPNG`。

单个 PDF：

```bash
python -m vlm_cadcoder.cli render-pdf \
  --pdf DataFlow/01.RawPDFWithSTEP/X350-05-070-A.pdf \
  --sample-id X350-05-070-A \
  --dataflow-root DataFlow \
  --dpi 600 \
  --skip-multipage
```

输出：

```text
DataFlow/02.RawPNG/X350-05-070-A/page_001_600dpi.png
DataFlow/02.RawPNG/X350-05-070-A/page_001_600dpi.meta.json
```

参数说明：

- `--pdf`：输入 PDF 路径；
- `--sample-id`：样本 ID，建议与 PDF 文件名 stem 一致；
- `--dpi`：渲染分辨率，当前推荐 600；
- `--skip-multipage`：如果 PDF 不是单页，则跳过该 PDF。

如果要渲染多页 PDF，去掉 `--skip-multipage`。

批量处理 `01.RawPDFWithSTEP` 下所有 PDF：

```bash
python -m vlm_cadcoder.cli render-pdf-batch \
  --raw-dir DataFlow/01.RawPDFWithSTEP \
  --dataflow-root DataFlow \
  --dpi 600 \
  --skip-multipage
```

批量处理并跳过已经存在的第一页 PNG：

```bash
python -m vlm_cadcoder.cli render-pdf-batch \
  --raw-dir DataFlow/01.RawPDFWithSTEP \
  --dataflow-root DataFlow \
  --dpi 600 \
  --skip-multipage \
  --skip-existing
```

如果 `01.RawPDFWithSTEP` 下还有子目录，也希望递归扫描 PDF：

```bash
python -m vlm_cadcoder.cli render-pdf-batch \
  --raw-dir DataFlow/01.RawPDFWithSTEP \
  --dataflow-root DataFlow \
  --dpi 600 \
  --skip-multipage \
  --recursive
```

批量命令输出示例：

```text
Rendered 20 PDFs / 20 pages; skipped 0; failed 0
[rendered] X350-05-070-A: 1 page(s)
```

参数说明：

- `--raw-dir`：输入 PDF 根目录；
- `--dataflow-root`：DataFlow 根目录；
- `--dpi`：渲染分辨率；
- `--skip-multipage`：遇到多页 PDF 时跳过该 PDF；
- `--skip-existing`：如果 `02.RawPNG/<sample_id>/page_001_<dpi>dpi.png` 已存在，则跳过；
- `--recursive`：递归扫描子目录中的 PDF；
- `--fail-fast`：任一 PDF 渲染失败时立即停止。

递归模式下，子目录 PDF 的 `sample_id` 会使用相对路径拼接，例如：

```text
DataFlow/01.RawPDFWithSTEP/testView2CAD/A.pdf
-> sample_id = testView2CAD__A
```

## 3. `02.RawPNG -> 03.LayoutAnalysis + 04.CleanPNG`

用途：对整张 PNG 做页面级 layout 分析，去除外边框、标题栏、孔表、版本表等非视图区信息，同时保留被移除区域 crop 供后续语义抽取。

命令：

```bash
python -m vlm_cadcoder.cli clean-layout \
  --image DataFlow/02.RawPNG/X350-05-070-A/page_001_600dpi.png \
  --sample-id X350-05-070-A \
  --page 1 \
  --dataflow-root DataFlow
```

输出：

```text
DataFlow/03.LayoutAnalysis/X350-05-070-A/page_001_layout.json
DataFlow/03.LayoutAnalysis/X350-05-070-A/page_001_overlay.png
DataFlow/03.LayoutAnalysis/X350-05-070-A/regions/*.png
DataFlow/04.CleanPNG/X350-05-070-A/page_001_clean.png
DataFlow/04.CleanPNG/X350-05-070-A/page_001_remove_mask.png
```

可选参数：

```bash
python -m vlm_cadcoder.cli clean-layout \
  --image DataFlow/02.RawPNG/X350-05-070-A/page_001_600dpi.png \
  --sample-id X350-05-070-A \
  --page 1 \
  --dataflow-root DataFlow \
  --output-stem page_001 \
  --no-save-crops \
  --no-save-overlay
```

说明：

- 默认会保存 removed-region crops 和 overlay；
- `04.CleanPNG` 保持原图尺寸，只白掉需要移除的区域，因此后续 bbox 坐标仍可与原图对齐。

## 4. `04.CleanPNG -> 05.ViewDetection`

状态：正式自动模块待接入。

目标：输入 clean page，输出每个 `view_with_annotations` 的 bbox。

计划输入：

```text
DataFlow/04.CleanPNG/<sample_id>/page_001_clean.png
```

计划输出：

```text
DataFlow/05.ViewDetection/<sample_id>/page_001_views.json
DataFlow/05.ViewDetection/<sample_id>/page_001_view_overlay.png
```

计划 JSON 结构：

```json
{
  "sample_id": "X350-05-070-A",
  "page": 1,
  "image_size": {"width": 6000, "height": 4000},
  "views": [
    {
      "view_id": "view_001",
      "label": "view_with_annotations",
      "bbox": [100, 200, 2500, 1800],
      "score": 0.93,
      "source": "sketchsegment_view_detector"
    }
  ]
}
```

推荐接入方案：

```text
SketchSegment ViewBlockDetector
-> 输出 view_with_annotations bbox
-> 转写为 DataFlow/05.ViewDetection/<sample_id>/page_001_views.json
```

当前临时替代方案：使用外部裁剪好的 `DataFlow/06.SingleViews/testView2CAD/` crops，跳过 `05.ViewDetection`。

## 5. `05.ViewDetection -> 06.SingleViews`

状态：正式自动裁剪模块待接入。

目标：根据 `05.ViewDetection` 的 bbox，从 clean page 裁剪单视图图块。

计划输入：

```text
DataFlow/04.CleanPNG/<sample_id>/page_001_clean.png
DataFlow/05.ViewDetection/<sample_id>/page_001_views.json
```

计划输出：

```text
DataFlow/06.SingleViews/<sample_id>/view_001/clean_view_with_annotations.png
DataFlow/06.SingleViews/<sample_id>/view_001/view_metadata.json
DataFlow/06.SingleViews/<sample_id>/view_002/clean_view_with_annotations.png
DataFlow/06.SingleViews/<sample_id>/view_002/view_metadata.json
```

建议每个 view 至少保留：

```text
raw_view_with_annotations.png       可选，从 raw page 裁剪
clean_view_with_annotations.png     必须，从 clean page 裁剪
geometry_core.png                   后续生成，尽量只含几何核心
view_metadata.json                  必须，记录 bbox、score、source、坐标系
annotations.json                    可选，PMI detector/OCR 输出
```

当前外部 crops 目录：

```text
DataFlow/06.SingleViews/testView2CAD/<sample_id>/cut-img/*.png
DataFlow/06.SingleViews/testView2CAD/<sample_id>/cut-json/*.json
```

这些外部 crops 可用于下游原型，但不能作为自动视图检测性能证据。

## 6. `06.SingleViews -> benchmark experiments`

用途：对 full page、clean page 或 single-view crop 运行小模型筛选任务。

### 6.1 单图任务

`view_count` 示例：

```bash
python -m vlm_cadcoder.benchmarks.model_screening.runner \
  --model qwen2_5_vl_3b \
  --task view_count \
  --image DataFlow/02.RawPNG/X350-05-070-A/page_001_600dpi.png \
  --output-root experiments/model_screening
```

`dimension_ocr` 示例：

```bash
python -m vlm_cadcoder.benchmarks.model_screening.runner \
  --model qwen2_5_vl_3b \
  --task dimension_ocr \
  --image DataFlow/06.SingleViews/testView2CAD/2023-2024-1-923/cut-img/2023-2024-1-923_crop_1.png \
  --output-root experiments/external_crops
```

`feature_count` 示例：

```bash
python -m vlm_cadcoder.benchmarks.model_screening.runner \
  --model qwen2_5_vl_3b \
  --task feature_count \
  --image DataFlow/06.SingleViews/testView2CAD/2023-2024-1-923/cut-img/2023-2024-1-923_crop_1.png \
  --output-root experiments/external_crops
```

输出：

```text
experiments/<output_root>/<timestamp>_<model>_<task>/predictions.jsonl
experiments/<output_root>/<timestamp>_<model>_<task>/metrics.json
experiments/<output_root>/<timestamp>_<model>_<task>/config.json
```

### 6.2 split 批量任务

如果已有 split 文件，例如：

```text
data/benchmark_small.jsonl
```

运行全部任务：

```bash
python -m vlm_cadcoder.benchmarks.model_screening.runner \
  --model qwen2_5_vl_3b \
  --split data/benchmark_small.jsonl \
  --dataflow-root DataFlow \
  --output-root experiments/model_screening
```

只运行某个任务：

```bash
python -m vlm_cadcoder.benchmarks.model_screening.runner \
  --model qwen2_5_vl_3b \
  --split data/benchmark_small.jsonl \
  --task feature_count \
  --dataflow-root DataFlow \
  --output-root experiments/model_screening
```

支持任务：

```text
view_count
view_classification
dimension_ocr
feature_count
json_stability
```

## 7. `06.SingleViews + experiments -> 10.StructuredCADRepresentation + 11 prompt`

用途：使用外部 single-view crops、clean 图、VLM benchmark 输出和 STEP 真值，生成最小 DrawingIR、建模计划和 CadQuery prompt。

当前该流程主要服务 `testView2CAD` 外部裁剪原型。

输入：

```text
DataFlow/06.SingleViews/testView2CAD/<sample_id>/cut-img/*.png
DataFlow/06.SingleViews/testView2CAD/<sample_id>/cut-json/*.json
DataFlow/04.CleanPNG/testView2CAD/<sample_id>.png
DataFlow/01.RawPDFWithSTEP/testView2CAD/<sample_id>.STEP
experiments/external_crops/**/predictions.jsonl
```

命令：

```bash
python -m vlm_cadcoder.cli build-view2cad-prototype \
  --sample-id 2023-2024-1-923 \
  --dataflow-root DataFlow \
  --external-crop-set testView2CAD \
  --experiments-root experiments/external_crops \
  --output-set testView2CAD
```

输出：

```text
DataFlow/10.StructuredCADRepresentation/testView2CAD/2023-2024-1-923/external_crop_manifest.json
DataFlow/10.StructuredCADRepresentation/testView2CAD/2023-2024-1-923/minimal_drawing_ir.json
DataFlow/10.StructuredCADRepresentation/testView2CAD/2023-2024-1-923/modeling_plan.json
DataFlow/11.CADProgram/testView2CAD/2023-2024-1-923/cadquery_generation_prompt.md
```

说明：这是外部 crops 原型闭环，不代表正式 DrawingIR 自动生成已经完成。

## 8. `10.StructuredCADRepresentation -> 11.CADProgram` 规则草稿

用途：基于 `minimal_drawing_ir.json` 和 `modeling_plan.json` 生成参数复核表和规则化 CadQuery 草稿脚本。

命令：

```bash
python -m vlm_cadcoder.cli build-cadquery-draft \
  --sample-id 2023-2024-1-923 \
  --dataflow-root DataFlow \
  --input-set testView2CAD \
  --output-set testView2CAD \
  --part-family rectangular_plate
```

输出：

```text
DataFlow/11.CADProgram/testView2CAD/2023-2024-1-923/cadquery_parameters.json
DataFlow/11.CADProgram/testView2CAD/2023-2024-1-923/cadquery_draft.py
```

说明：

- `cadquery_draft.py` 是规则化 baseline/scaffold；
- 参数表中标记为 `needs_review` 的字段不能视为最终图纸约束；
- 该草稿可用于分析图纸理解结果是否足够支撑建模，而不是最终 CAD 生成方法。

## 9. `10/11 prompt -> 11.CADProgram` LLM 生成 CadQuery

用途：让服务器上的 VLM/LLM 基于 `cadquery_generation_prompt.md`、clean 图和 single-view crops 直接生成 CadQuery 代码。

命令：

```bash
python -m vlm_cadcoder.cli generate-cadquery-llm \
  --sample-id 2023-2024-1-923 \
  --model qwen2_5_vl_3b \
  --dataflow-root DataFlow \
  --model-config configs/models.json \
  --input-set testView2CAD \
  --output-set testView2CAD \
  --max-new-tokens 4096
```

输出：

```text
DataFlow/11.CADProgram/testView2CAD/2023-2024-1-923/cadquery_llm_generated.raw.md
DataFlow/11.CADProgram/testView2CAD/2023-2024-1-923/cadquery_llm_generated.py
```

注意：

- 当前直接 VLM/LLM 生成 CadQuery 属于 baseline/failure probe；
- 如果生成脚本出现 API 幻觉，不建议无限修 prompt；
- 主线仍应回到 DrawingIR、尺寸-几何绑定和约束图。

## 10. CadQuery LLM 输出后处理

用途：清理已有 LLM 输出中的 markdown fence、错误 import、导出语句等格式问题。

命令：

```bash
python -m vlm_cadcoder.cli sanitize-cadquery-llm \
  --input DataFlow/11.CADProgram/testView2CAD/2023-2024-1-923/cadquery_llm_generated.py
```

指定输出文件：

```bash
python -m vlm_cadcoder.cli sanitize-cadquery-llm \
  --input DataFlow/11.CADProgram/testView2CAD/2023-2024-1-923/cadquery_llm_generated.raw.md \
  --output DataFlow/11.CADProgram/testView2CAD/2023-2024-1-923/cadquery_llm_generated.py
```

## 11. CadQuery 脚本执行

用途：在服务器 CadQuery 环境中执行生成脚本并导出 STEP。

规则草稿：

```bash
python DataFlow/11.CADProgram/testView2CAD/2023-2024-1-923/cadquery_draft.py
```

LLM 生成脚本：

```bash
python DataFlow/11.CADProgram/testView2CAD/2023-2024-1-923/cadquery_llm_generated.py
```

常见输出：

```text
DataFlow/11.CADProgram/testView2CAD/2023-2024-1-923/2023-2024-1-923_cadquery_draft.step
```

说明：执行成功只代表脚本语法/API 可运行，不代表几何与图纸一致。后续仍需 STEP/渲染/尺寸约束校验模块。

## 12. 建议的一次性运行顺序

以 `X350-05-070-A` 为例，从 PDF 到 clean page：

```bash
export PYTHONPATH=src

python -m vlm_cadcoder.cli build-sample-index \
  --raw-dir DataFlow/01.RawPDFWithSTEP \
  --output data/samples.csv

python -m vlm_cadcoder.cli render-pdf \
  --pdf DataFlow/01.RawPDFWithSTEP/X350-05-070-A.pdf \
  --sample-id X350-05-070-A \
  --dataflow-root DataFlow \
  --dpi 600 \
  --skip-multipage

python -m vlm_cadcoder.cli clean-layout \
  --image DataFlow/02.RawPNG/X350-05-070-A/page_001_600dpi.png \
  --sample-id X350-05-070-A \
  --page 1 \
  --dataflow-root DataFlow
```

以 `2023-2024-1-923` 为例，从外部 crops 到 CadQuery 原型：

```bash
export PYTHONPATH=src

python -m vlm_cadcoder.benchmarks.model_screening.runner \
  --model qwen2_5_vl_3b \
  --task dimension_ocr \
  --image DataFlow/06.SingleViews/testView2CAD/2023-2024-1-923/cut-img/2023-2024-1-923_crop_1.png \
  --output-root experiments/external_crops

python -m vlm_cadcoder.benchmarks.model_screening.runner \
  --model qwen2_5_vl_3b \
  --task feature_count \
  --image DataFlow/06.SingleViews/testView2CAD/2023-2024-1-923/cut-img/2023-2024-1-923_crop_1.png \
  --output-root experiments/external_crops

python -m vlm_cadcoder.cli build-view2cad-prototype \
  --sample-id 2023-2024-1-923 \
  --dataflow-root DataFlow \
  --experiments-root experiments/external_crops

python -m vlm_cadcoder.cli build-cadquery-draft \
  --sample-id 2023-2024-1-923 \
  --dataflow-root DataFlow
```

## 13. 后续需要补的命令

建议下一步补齐以下 CLI：

```text
clean-layout-batch
detect-views
export-single-views
classify-views
extract-view-features
build-drawing-ir
validate-cadquery-step
```

其中优先级最高的是：

```text
detect-views
export-single-views
```

这两个命令完成后，正式链路就可以从：

```text
01 -> 02 -> 03/04 -> 05 -> 06
```

稳定推进到：

```text
06 -> OCR/VLM -> 10 -> 11
```
