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
| `02 -> 03/04` | 已实现 | 单图或批量页面级 layout 分析与 clean PNG，包含表格、边框、内图框条带和角落元信息框处理 |
| `04 -> 05` | 已形成联动流程 | SketchSegment 生成 view candidates，LLM-CADCoder 使用 `ViewCandidateFilter` 后置过滤 |
| `05 -> 06` | 已形成联动流程 | 由 SketchSegment 导出脚本根据过滤后的 view bbox 裁剪单视图 |
| `05/06 -> audit` | 已实现 | 审计 view detection 与 single-view crops 是否一致 |
| `06 -> geometry_core` | 已实现调用器 | 调用外部 SketchPic2ViewPic U-Net，把标注视图净化为几何核心图 |
| `geometry_core -> repair` | 已实现 MVP | 对 U-Net 几何核心图做轻量水平/竖直断线桥接和小碎片过滤；作为 sidecar 输出，不覆盖原图 |
| `geometry_core -> primitive repair` | 已实现 MVP | 保守生成 circle_arc 基元修复候选，line 需显式开启；拒绝孤立闭合矩形框 |
| `geometry_core -> audit` | 已实现 | 对 geometry core 进行质量分层，输出 CSV/JSON/contact sheet |
| `06 -> 07` | 已实现基线 | 根据 view bbox 几何和页面位置生成启发式视图类型 baseline |
| `06 -> benchmark` | 已实现 | 使用 single-view crops 跑 VLM 小模型任务，支持从 DrawingIR 批量运行正式非轴测视图 |
| `07 + geometry_core audit -> 10` | 已实现初版 | 从 05/06/07 生成 DrawingIR，并把 A 类 geometry_core 转为低层几何候选 |
| `10 -> 08` | 已实现初版 | 把 `geometry_component` 低层候选提升为保守语义特征候选，供人工复核和后续约束绑定 |
| `10 + dimension_ocr -> 08` | 已实现 MVP | 读取 DrawingIR 和 VLM/OCR `dimension_ocr` 预测，生成可复核尺寸候选 |
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

用途：对整张 PNG 做页面级 layout 分析，去除外边框、内图框条带、角落元信息框、标题栏、孔表、版本表、技术要求等非视图区信息，同时保留有语义价值的被移除区域 crop 供后续语义抽取。

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
- 当前 layout cleaner 已加入轻量增强，可处理靠近页边的 `inner_frame_strip` 和左上/右上稀疏 `corner_metadata_box`；
- `04.CleanPNG` 保持原图尺寸，只白掉需要移除的区域，因此后续 bbox 坐标仍可与原图对齐。

批量处理 `02.RawPNG` 下所有已渲染页面：

```bash
python -m vlm_cadcoder.cli clean-layout-batch \
  --raw-png-dir DataFlow/02.RawPNG \
  --dataflow-root DataFlow \
  --dpi 600
```

批量处理并跳过已经存在的 clean page：

```bash
python -m vlm_cadcoder.cli clean-layout-batch \
  --raw-png-dir DataFlow/02.RawPNG \
  --dataflow-root DataFlow \
  --dpi 600 \
  --skip-existing
```

如果只想生成 clean page 和 mask，不保存 overlay 或 removed-region crops：

```bash
python -m vlm_cadcoder.cli clean-layout-batch \
  --raw-png-dir DataFlow/02.RawPNG \
  --dataflow-root DataFlow \
  --dpi 600 \
  --no-save-crops \
  --no-save-overlay
```

批量命令输出示例：

```text
Cleaned 20 page(s); skipped 0; failed 0
[cleaned] X350-05-070-A page 001: 4 removable region(s)
```

参数说明：

- `--raw-png-dir`：输入 RawPNG 根目录，默认 `DataFlow/02.RawPNG`；
- `--dataflow-root`：DataFlow 根目录；
- `--dpi`：只处理匹配 `page_*_<dpi>dpi.png` 的页面；
- `--skip-existing`：如果 `04.CleanPNG/<sample_id>/page_<n>_clean.png` 已存在，则跳过；
- `--no-save-crops`：不保存被移除表格/边框区域的 crops；
- `--no-save-overlay`：不保存 layout overlay；
- `--fail-fast`：任一页面处理失败时立即停止。

批量命令会从目录结构推断 `sample_id` 和页码：

```text
DataFlow/02.RawPNG/X350-05-070-A/page_001_600dpi.png
-> sample_id = X350-05-070-A
-> page = 1
-> output_stem = page_001
```

## 4. `04.CleanPNG -> 05.ViewDetection`

状态：已形成联动流程。SketchSegment 检测模型作为 raw candidate 来源，LLM-CADCoder 提供 `ViewCandidateFilter` 后置过滤命令。

目标：输入 clean page，输出每个 `view_with_annotations` 的 bbox。

输入：

```text
DataFlow/04.CleanPNG/<sample_id>/page_001_clean.png
DataFlow/05.ViewDetection/<sample_id>/page_001_views.json       # SketchSegment raw detections
```

输出：

```text
DataFlow/05.ViewDetection/<sample_id>/page_001_views_raw.json
DataFlow/05.ViewDetection/<sample_id>/page_001_views.json              # filtered detections
DataFlow/05.ViewDetection/<sample_id>/page_001_rejected_views.json
DataFlow/05.ViewDetection/<sample_id>/page_001_view_filter_overlay.png
```

过滤后的 JSON 结构：

```json
{
  "sample_id": "X350-05-070-A",
  "page": 1,
  "image_size": {"width": 6000, "height": 4000},
  "views": [
    {
      "view_id": "view_001",
      "source_view_id": "view_003",
      "label": "view_with_annotations",
      "bbox": [100, 200, 2500, 1800],
      "score": 0.93,
      "source": "sketchsegment_view_detector",
      "filter": {
        "accepted": true,
        "reject_reasons": [],
        "source": "view_candidate_filter_v1"
      }
    }
  ]
}
```

单个样本过滤命令：

```bash
export PYTHONPATH=src

python -m vlm_cadcoder.cli filter-view-detections \
  --sample-id M001-08-006-B \
  --dataflow-root DataFlow
```

批量过滤命令：

```bash
export PYTHONPATH=src

python -m vlm_cadcoder.cli filter-view-detections-batch \
  --dataflow-root DataFlow
```

批量过滤时会严格匹配 `page_<number>_views.json` 和 `page_<number>_views_raw.json`，不会把 `page_001_rejected_views.json` 当作输入。

如果只想调试，不覆盖原始 `05.ViewDetection/<sample_id>/page_001_views.json`，可以写到临时目录：

```bash
export PYTHONPATH=src

python -m vlm_cadcoder.cli filter-view-detections \
  --sample-id M001-08-006-B \
  --dataflow-root DataFlow \
  --output-json .tmp/view_filter/M001-08-006-B/page_001_views.json
```

关键参数：

```text
--min-score                  默认 0.5；低分候选若具有细线视图特征，仍允许保留
--top-strip-score            默认 0.6；顶部短条低于该分数时拒绝
--dense-ink-ratio            默认 0.16；用于拒绝“1套”等粗黑大字块
--dense-thick-ink-ratio      默认 0.14；用于拒绝粗黑连通文本块
--no-save-overlay            不保存过滤可视化 overlay
```

接入方案：

```text
SketchSegment ViewBlockDetector
-> 输出 view_with_annotations bbox
-> 转写为 DataFlow/05.ViewDetection/<sample_id>/page_001_views.json
-> LLM-CADCoder ViewCandidateFilter
-> 输出 filtered views + rejected views
```

注意：`05.ViewDetection/<sample_id>/page_001_views.json` 应作为 `06.SingleViews` 的唯一正式来源。旧版、未过滤或 `copy` 样本建议隔离为失败分析/消融样本，不要混入正式评测。

## 5. `05.ViewDetection -> 06.SingleViews`

状态：由 SketchSegment 项目导出脚本生成，当前项目已形成 `01 -> 06` 第一版数据流闭环。

目标：根据 `05.ViewDetection` 的 bbox，从 clean page 裁剪单视图图块。

输入：

```text
DataFlow/04.CleanPNG/<sample_id>/page_001_clean.png
DataFlow/05.ViewDetection/<sample_id>/page_001_views.json
```

输出：

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
geometry_core.png                   由 SketchPic2ViewPic U-Net 生成，尽量只含几何核心
view_metadata.json                  必须，记录 bbox、score、source、坐标系
annotations.json                    可选，PMI detector/OCR 输出
```

在 SketchSegment 项目中运行：

```bash
  python scripts/09_export_llmcad_views.py \
    --clean-image "$img" \
    --views-json "/home/zxwcax/PycharmProjects/LLMCAD-coder/DataFlow/05.ViewDetection/$sample/page_001_views.json" \
    --out-dir "/home/zxwcax/PycharmProjects/LLMCAD-coder/DataFlow/06.SingleViews" \
    --weights runs/detect/runs/view/exp_all/weights/best.pt \
    --sample-id "$sample"
```

其中 `--views-json` 建议使用经过 `ViewCandidateFilter` 过滤后的 `page_001_views.json`，这样 `06.SingleViews` 不会继续导出已拒绝的候选框。

当前外部 crops 目录：

```text
DataFlow/06.SingleViews/testView2CAD/<sample_id>/cut-img/*.png
DataFlow/06.SingleViews/testView2CAD/<sample_id>/cut-json/*.json
```

这些外部 crops 可用于下游原型，但不能作为自动视图检测性能证据。

### 5.1 `05.ViewDetection + 06.SingleViews -> audit`

用途：审计 `05.ViewDetection` 与 `06.SingleViews` 是否一致，生成后续人工验收表的基础版本。

命令：

```bash
export PYTHONPATH=src

python -m vlm_cadcoder.cli audit-single-views \
  --dataflow-root DataFlow
```

输出：

```text
DataFlow/06.SingleViews/audit_single_views.csv
DataFlow/06.SingleViews/audit_single_views.json
```

CSV/JSON 中会记录：

```text
sample_id
page
official_candidate
has_view_detection
has_single_views
detected_view_count
rejected_view_count
exported_view_count
metadata_count
clean_image_count
is_05_06_consistent
needs_manual_review
review_reasons
view_detection_path
single_views_dir
```

如果要指定输出路径：

```bash
python -m vlm_cadcoder.cli audit-single-views \
  --dataflow-root DataFlow \
  --output-csv DataFlow/06.SingleViews/audit_single_views.csv \
  --output-json DataFlow/06.SingleViews/audit_single_views.json
```

解释：

- `is_05_06_consistent=true` 表示 `05` 中 accepted views 数量、`06` 中 view 目录数量、metadata 数量和 clean image 数量一致；
- `needs_manual_review=true` 表示样本需要人工检查；
- 常见 `review_reasons` 包括 `view_count_mismatch`、`missing_view_detection`、`missing_single_views`、`copy_sample`、`missing_view_metadata`、`missing_clean_view_image`；
- `copy`、旧版或未过滤样本会被标记为非正式候选，建议隔离为失败分析或消融样本。

### 5.2 `06.SingleViews -> geometry_core.png`

状态：已接入外部调用器。该阶段使用独立项目 `SketchPic2ViewPic` 中当前选定的 U-Net 模型，将每个 `clean_view_with_annotations.png` 净化为尽量只包含几何轮廓的 `geometry_core.png`。

外部项目不复制到 LLM-CADCoder。本项目只负责扫描 `06.SingleViews`、调用外部推理命令、回填统一命名的结果文件。

当前采用的外部配置：

```text
SketchPic2ViewPic/configs/unet_baseline.yaml
run_name: unet_tversky_a07_b03
checkpoint: runs/unet_tversky_a07_b03/checkpoints/best.pt
```

输入：

```text
DataFlow/06.SingleViews/<sample_id>/view_001/clean_view_with_annotations.png
```

输出：

```text
DataFlow/06.SingleViews/<sample_id>/view_001/geometry_core.png
DataFlow/06.SingleViews/<sample_id>/view_001/geometry_core_mask.png
DataFlow/06.SingleViews/<sample_id>/view_001/geometry_core_prob.png
DataFlow/06.SingleViews/<sample_id>/view_001/geometry_core.meta.json
DataFlow/06.SingleViews/<sample_id>/view_001/geometry_core_unet/*.png
```

建议先确认外部项目环境：

```bash
cd /home/zxwcax/Projects/SketchPic2View/SketchPic2ViewPic
conda activate sketchpic2viewpic
python -m pip install -e .

python -m sketchpic2viewpic infer \
  --config configs/unet_baseline.yaml \
  --checkpoint runs/unet_tversky_a07_b03/checkpoints/best.pt \
  --input /home/zxwcax/PycharmProjects/LLMCAD-coder/DataFlow/06.SingleViews/M001-08-006-B/view_001/clean_view_with_annotations.png \
  --output /home/zxwcax/PycharmProjects/LLMCAD-coder/DataFlow/06.SingleViews/M001-08-006-B/view_001/geometry_core_unet
```

上面的外部命令会生成：

```text
geometry_core_unet/clean_view_with_annotations_clean.png
geometry_core_unet/clean_view_with_annotations_mask.png
geometry_core_unet/clean_view_with_annotations_prob.png
```

正式推荐使用 LLM-CADCoder wrapper 自动回填文件名。

先 dry-run 检查将要执行的外部命令：

```bash
cd /home/zxwcax/PycharmProjects/LLMCAD-coder
export PYTHONPATH=src
export SKETCHPIC2VIEWPIC_ROOT=/home/zxwcax/Projects/SketchPic2View/SketchPic2ViewPic

python -m vlm_cadcoder.cli generate-geometry-core-unet \
  --sample-id M001-08-006-B \
  --dataflow-root DataFlow \
  --python /home/zxwcax/anaconda3/envs/sketchpic2viewpic/bin/python \
  --dry-run
```

处理单个样本：

```bash
python -m vlm_cadcoder.cli generate-geometry-core-unet \
  --sample-id M001-08-006-B \
  --dataflow-root DataFlow \
  --sketchpic2viewpic-root /home/zxwcax/Projects/SketchPic2View/SketchPic2ViewPic \
  --python /home/zxwcax/anaconda3/envs/sketchpic2viewpic/bin/python \
  --skip-existing
```

批量处理所有正式样本：

```bash
python -m vlm_cadcoder.cli generate-geometry-core-unet \
  --dataflow-root DataFlow \
  --sketchpic2viewpic-root /home/zxwcax/Projects/SketchPic2View/SketchPic2ViewPic \
  --python /home/zxwcax/anaconda3/envs/sketchpic2viewpic/bin/python \
  --skip-existing
```

默认推理覆盖参数与 `configs/unet_baseline.yaml` 保持一致，并显式传入：

```text
inference.resize_max_side=3072
inference.resize_min_side=1024
inference.patch_size=768
inference.stride=576
inference.threshold=0.85
```

如果临时调整阈值：

```bash
python -m vlm_cadcoder.cli generate-geometry-core-unet \
  --sample-id M001-08-006-B \
  --dataflow-root DataFlow \
  --sketchpic2viewpic-root /home/zxwcax/Projects/SketchPic2View/SketchPic2ViewPic \
  --python /home/zxwcax/anaconda3/envs/sketchpic2viewpic/bin/python \
  --override inference.threshold=0.9
```

说明：

- wrapper 默认跳过 `*-copy` 样本和 `testView2CAD` 外部原型目录；
- 如果 `geometry_core.png` 已存在，加 `--skip-existing` 会跳过该 view；
- 如果只想看外部命令，不执行推理，加 `--dry-run`；
- `geometry_core.png` 用于后续特征识别、几何约束识别和视图间对应；`clean_view_with_annotations.png` 仍保留尺寸、PMI、引线等语义信息，不能被覆盖。

### 5.3 `geometry_core -> repair`

用途：对 `geometry_core.png` 做轻量拓扑修复，优先处理 U-Net 去标注后出现的水平/竖直小断线，并移除极小碎片。该命令不会覆盖原始 `geometry_core.png`，而是输出独立的 repaired 版本，便于后续做消融对比。

输入：

```text
DataFlow/06.SingleViews/<sample_id>/view_001/geometry_core.png
DataFlow/06.SingleViews/<sample_id>/view_001/clean_view_with_annotations.png  # 可选，仅记录证据链
DataFlow/06.SingleViews/<sample_id>/view_001/geometry_core_prob.png           # 可选，仅记录证据链
```

输出：

```text
DataFlow/06.SingleViews/<sample_id>/view_001/geometry_core_repaired.png
DataFlow/06.SingleViews/<sample_id>/view_001/geometry_core_repair_overlay.png
DataFlow/06.SingleViews/<sample_id>/view_001/geometry_core_repair.meta.json
DataFlow/06.SingleViews/geometry_core_repair_summary.csv
DataFlow/06.SingleViews/geometry_core_repair_summary.json
```

单个样本：

```bash
export PYTHONPATH=src

python -m vlm_cadcoder.cli repair-geometry-core \
  --sample-id M001-08-006-B \
  --dataflow-root DataFlow \
  --skip-existing
```

批量处理正式样本：

```bash
python -m vlm_cadcoder.cli repair-geometry-core \
  --dataflow-root DataFlow \
  --skip-existing
```

常用参数：

```text
--max-gap-px          允许桥接的最大断线间隙，默认 12
--min-segment-px      gap 两侧最短线段长度，默认 16
--bridge-support-radius  搜索相邻扫描线支持的半径，默认 1
--min-bridge-support  至少需要多少条相邻扫描线支持同一 gap，默认 2
--tiny-area-px        小于该像素面积的连通域会被移除，默认 12
--direction           可重复传入 horizontal/vertical；不传时两者都启用
--dry-run             只扫描并生成汇总，不写 repaired/overlay/meta
--fail-fast           任一 view 失败后立即停止
```

说明：

- repaired 图当前是后处理候选，不自动替换 DrawingIR 中的 `geometry_core.png` 输入；
- overlay 中黑色为原始保留墨迹，红色为新增桥接像素，灰色为被移除碎片；
- 默认桥接要求相邻扫描线也存在相近 gap，单行/单列孤证不会桥接；若要做更激进的消融，可把 `--min-bridge-support` 调为 1；
- 该 MVP 只做保守的水平/竖直小间隙修复，斜线、圆弧和基于概率图的弱线证据可作为下一阶段增强；
- `--skip-existing` 只有在 repaired、overlay 和 metadata 都存在时才跳过，避免保留半成品输出。

### 5.4 `geometry_core -> primitive repair candidates`

用途：对 `geometry_core.png` 生成保守几何基元修复候选。该阶段只把用于 gap bridging 的 `line` 与 `circle_arc` 写入 accepted candidates；孤立闭合矩形框会写入 rejected candidates，避免把标注框、局部视图框或文字框误保留为 CAD 几何。

输入：

```text
DataFlow/06.SingleViews/<sample_id>/view_001/geometry_core.png
```

输出：

```text
DataFlow/06.SingleViews/<sample_id>/view_001/geometry_core_primitive_repaired.png
DataFlow/06.SingleViews/<sample_id>/view_001/primitive_repair_overlay.png
DataFlow/06.SingleViews/<sample_id>/view_001/primitive_candidates.json
DataFlow/06.SingleViews/geometry_primitive_repair_summary.csv
DataFlow/06.SingleViews/geometry_primitive_repair_summary.json
```

单个样本：

```bash
export PYTHONPATH=src

python -m vlm_cadcoder.cli repair-geometry-primitives \
  --sample-id M001-08-006-B \
  --dataflow-root DataFlow \
  --skip-existing
```

批量处理正式样本：

```bash
python -m vlm_cadcoder.cli repair-geometry-primitives \
  --dataflow-root DataFlow \
  --skip-existing
```

常用参数：

```text
--primitive-type                 可重复传入 line/circle_arc；不传时默认只启用 circle_arc
--max-line-gap-px                允许桥接的水平/竖直直线最大断口，默认 12
--min-line-segment-px            直线 gap 两侧最短线段长度，默认 16
--min-existing-arc-coverage      接受圆弧候选所需的最小现有弧线覆盖率，默认 0.55
--min-circle-gap-ratio           接受圆弧候选所需的最小缺口比例，默认 0.05
--max-circle-gap-ratio           接受圆弧候选允许的最大缺口比例，默认 0.35
--circle-radius-tolerance        圆弧拟合平均径向误差阈值，默认 0.18
--dry-run                        只扫描并生成汇总，不写 view 级输出
```

说明：

- 该命令是 candidate artifact，不自动替换 DrawingIR 或 CAD 生成输入；
- overlay 中黑色为原始墨迹，红色为基元候选新增修复像素；
- `primitive_candidates.json` 同时记录 accepted 和 rejected candidates，后续可用于人工复核、消融和规则迭代；
- 直线候选与上一层 `repair-geometry-core` 有重叠，默认关闭；如需消融可显式传入 `--primitive-type line --primitive-type circle_arc`；
- 矩形框默认只拒绝不补全，后续若要处理真实矩形外轮廓，需要结合 view 拓扑、文本区域和尺寸绑定证据再开放。

### 5.5 `geometry_core -> quality audit`

用途：对 `06.SingleViews` 中的 `geometry_core.png` 做质量审计，避免把 U-Net 净化失败的视图直接送入后续特征识别和约束图构建。

默认审计范围会结合 `07.ViewClassification`：

```text
只审计 07/views 中的正式 view
跳过 07/skipped_views，也就是 05/07 已 reject 的 crop
默认跳过 type=isometric 的轴测图
如果样本缺少 07 分类文件，默认不审计该样本
```

原因：当前 `geometry_core` 主要服务后续特征识别、尺寸-几何绑定和正投影视图约束推理；rejected crop 和 isometric view 暂不进入正式几何质量统计。

审计输入：

```text
DataFlow/06.SingleViews/<sample_id>/view_001/clean_view_with_annotations.png
DataFlow/06.SingleViews/<sample_id>/view_001/geometry_core.png
DataFlow/06.SingleViews/<sample_id>/view_001/geometry_core_mask.png
DataFlow/06.SingleViews/<sample_id>/view_001/geometry_core_prob.png
DataFlow/07.ViewClassification/<sample_id>/page_001_view_classification.json
```

审计输出：

```text
DataFlow/06.SingleViews/geometry_core_audit.csv
DataFlow/06.SingleViews/geometry_core_audit.json
DataFlow/06.SingleViews/geometry_core_audit_contact_sheet.png
```

可选人工修正输入：

```text
DataFlow/06.SingleViews/geometry_core_audit_overrides.json
```

批量审计正式样本：

```bash
export PYTHONPATH=src

python -m vlm_cadcoder.cli audit-geometry-core \
  --dataflow-root DataFlow
```

只审计单个样本：

```bash
python -m vlm_cadcoder.cli audit-geometry-core \
  --sample-id M001-08-006-B \
  --dataflow-root DataFlow
```

如果要把轴测图也纳入审计：

```bash
python -m vlm_cadcoder.cli audit-geometry-core \
  --sample-id M001-08-006-B \
  --dataflow-root DataFlow \
  --include-isometric
```

如果要绕过 `07.ViewClassification`，直接审计 `06.SingleViews` 中所有 view 目录：

```bash
python -m vlm_cadcoder.cli audit-geometry-core \
  --dataflow-root DataFlow \
  --no-use-view-classification
```

该模式只用于排查或消融，因为它会重新纳入 rejected crop 和 isometric view。

如果不想生成 contact sheet：

```bash
python -m vlm_cadcoder.cli audit-geometry-core \
  --dataflow-root DataFlow \
  --no-save-contact-sheet
```

如果希望限制 contact sheet 中的样本数量：

```bash
python -m vlm_cadcoder.cli audit-geometry-core \
  --dataflow-root DataFlow \
  --contact-sheet-limit 80
```

CSV 中预留人工复核列：

```text
manual_quality_label
manual_notes
```

这些列用于记录当前审计结果中的人工覆盖信息。为了保证审计可复现，不建议直接手改 `geometry_core_audit.csv` 后作为长期依据；建议把人工结论写入：

```text
DataFlow/06.SingleViews/geometry_core_audit_overrides.json
```

示例：

```json
{
  "schema": "geometry_core_audit_overrides",
  "version": "0.1.0",
  "overrides": [
    {
      "sample_id": "M001-08-006-B",
      "view_id": "view_005",
      "exclude": true,
      "reason": "manual_excluded_stale_view"
    },
    {
      "sample_id": "X468-02-049-A",
      "view_id": "view_001",
      "quality_tier": "C",
      "reason": "manual_bad_geometry_core"
    },
    {
      "sample_id": "X476-04-002-A",
      "view_id": "view_001",
      "quality_tier": "A",
      "reason": "manual_good_geometry_core"
    }
  ]
}
```

如需指定其他人工修正文件：

```bash
python -m vlm_cadcoder.cli audit-geometry-core \
  --dataflow-root DataFlow \
  --overrides-json DataFlow/06.SingleViews/geometry_core_audit_overrides.json
```

建议人工标签使用：

```text
A/good       几何轮廓完整，标注基本去除，可作为几何主输入
B/usable     局部断线、碎片、残留或过度简化，只能作为辅助输入
C/bad        缺失、误删严重、伪线严重或不可用，应从后续自动特征抽取中排除
```

当前自动质量分层只作为初筛，不作为最终真值。它主要依据：

```text
geometry_core 是否存在
clean 与 geometry_core 尺寸是否一致
黑像素比例
geometry/clean 墨迹保留比例
明显额外墨迹比例
geometry 连通域碎片数量
mask/prob 输出是否完整
```

重要说明：

- `A` 表示机器规则暂未发现明显异常，不等于人工确认的高质量真值；
- `B/C` 必须优先人工复核；
- 该审计结果应作为后续 `08.Multi-viewFeatureExtraction` 的输入质量门控和失败分析来源；
- contact sheet 左侧为 `clean_view_with_annotations.png`，右侧为 `geometry_core.png`，用于快速人工判定。

## 6. `06.SingleViews -> 07.ViewClassification`

用途：对 `06.SingleViews` 中的单视图 crop 生成第一版视图类型 baseline。当前实现为启发式规则，不依赖 VLM，会先用 `05.ViewDetection/<sample_id>/page_001_views.json` 中的 accepted views 过滤 `06` 中的 crop，再根据 view bbox 面积、宽高比和页面位置判断：

```text
front
top
left
isometric
unknown
```

该结果用于人工复核、后续 VLM 分类对比和 DrawingIR 最小原型，不应视为最终标注真值。若 `06` 中仍残留 rejected crop，但它没有匹配到 `05` 中的 accepted bbox，会被写入 `skipped_views`，不会进入 `views` 分类结果。

批量分类正式样本，默认跳过 `-copy` 样本：

```bash
export PYTHONPATH=src

python -m vlm_cadcoder.cli classify-views \
  --dataflow-root DataFlow
```

单个样本：

```bash
python -m vlm_cadcoder.cli classify-views \
  --sample-id 32HA1252-008-040 \
  --dataflow-root DataFlow
```

如果需要同时处理 `-copy` 样本：

```bash
python -m vlm_cadcoder.cli classify-views \
  --dataflow-root DataFlow \
  --include-copy
```

输出：

```text
DataFlow/07.ViewClassification/<sample_id>/page_001_view_classification.json
DataFlow/07.ViewClassification/view_classification_summary.csv
DataFlow/07.ViewClassification/view_classification_summary.json
```

单样本 JSON 中每个 view 会包含：

```text
view_id
type
confidence
is_primary
needs_manual_review
reasons
bbox_on_page
crop_size
detector_score
image_clean
```

同时会记录：

```text
input_filter                  说明 07 使用了 05 accepted views 过滤 06 crops
skipped_views                 记录 06 中未匹配 05 accepted bbox 的 crop
```

说明：

- `front` 当前由最大非轴测视图启发式确定，但仍是主视图候选，不是人工真值；
- `top` 当前主要对应细长水平轮廓视图；
- `left` 当前主要对应细长竖向或右侧轮廓视图；
- `isometric` 当前只在存在三个及以上 accepted views 且几何位置符合右下斜视候选时给出；
- `needs_manual_review=true` 的结果需要人工复核，尤其是 `left/top/right` 具体投影方向。

## 7. `06.SingleViews -> benchmark experiments`

用途：对 full page、clean page 或 single-view crop 运行小模型筛选任务。

### 7.1 单图任务

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

### 7.2 split 批量任务

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

## 8. `07.ViewClassification -> 10.StructuredCADRepresentation`

用途：读取 `05.ViewDetection` accepted views、`06.SingleViews` crop metadata/image、`07.ViewClassification` 视图类型候选和 `06.SingleViews/geometry_core_audit.json`，生成正式链路的 DrawingIR v0.1。

该阶段已经接入 geometry_core 质量门控：只有 `quality_tier=A` 且 `geometry_core.png` 存在的正式视图会进入低层几何候选抽取；B/C、缺失 audit 记录或缺失图片的视图会进入 `quality.blocking_items`。当前候选仍是 `geometry_component` 级别的连通域 bbox，不等同于孔、槽、倒角、圆角等最终 CAD 语义特征。

批量生成正式样本，默认跳过 `-copy` 样本：

```bash
export PYTHONPATH=src

python -m vlm_cadcoder.cli build-drawing-ir \
  --dataflow-root DataFlow
```

显式指定 geometry_core 审计文件：

```bash
python -m vlm_cadcoder.cli build-drawing-ir \
  --dataflow-root DataFlow \
  --geometry-core-audit DataFlow/06.SingleViews/geometry_core_audit.json
```

只生成视图级 IR，不抽取低层几何候选：

```bash
python -m vlm_cadcoder.cli build-drawing-ir \
  --dataflow-root DataFlow \
  --no-extract-feature-candidates
```

单个样本：

```bash
python -m vlm_cadcoder.cli build-drawing-ir \
  --sample-id M001-08-006-B \
  --dataflow-root DataFlow
```

如果希望遇到坏 JSON 或缺失输入立即停止：

```bash
python -m vlm_cadcoder.cli build-drawing-ir \
  --dataflow-root DataFlow \
  --fail-fast
```

输出：

```text
DataFlow/10.StructuredCADRepresentation/<sample_id>/drawing_ir.json
DataFlow/10.StructuredCADRepresentation/drawing_ir_summary.csv
DataFlow/10.StructuredCADRepresentation/drawing_ir_summary.json
```

`drawing_ir.json` 主要包含：

```text
sheet                     图纸页级信息和 05/06/07 来源路径
views                     视图 bbox、crop、分类候选、置信度、人工复核标记、geometry_core 质量块
dimensions                当前为空，后续由 08/维度 OCR 填充
feature_candidates        A 类 geometry_core 产生的低层 geometry_component bbox，语义仍为 unclassified
constraints               当前为空，后续由 09/约束图构建填充
view_relations            当前为空，后续记录多视图投影/对应关系
skipped_views             07 中未进入正式 views 的 rejected/unmatched crops
provenance                构建器、分类器、过滤器等追溯信息
quality                   是否需要人工复核、是否可进入特征抽取、CAD 生成阻塞项
```

注意：

- `feature_candidates` 目前只表示干净几何视图中的墨迹连通域候选，用于给 08 阶段提供可复核的低层几何输入；
- 不应把 `geometry_component` 直接当作孔、槽、沉孔、倒角或圆角；
- 若 `quality.ready_for_feature_extraction=false`，说明该样本至少有一个正式视图被 geometry_core 质量门控挡住，应先复核 `geometry_core_audit_contact_sheet.png` 或更新 `geometry_core_audit_overrides.json`。

注意：该命令会严格解析 `07.ViewClassification/<sample_id>/page_001_view_classification.json`。如果 JSON 文件后面残留多余内容，会报 `Trailing data after JSON document`，需要先重新运行对应样本的 `classify-views`。

### 8.1 `10.StructuredCADRepresentation -> 08.Multi-viewFeatureExtraction`

用途：读取 `DataFlow/10.StructuredCADRepresentation/<sample_id>/drawing_ir.json` 中的低层 `geometry_component` 候选，生成 `08.Multi-viewFeatureExtraction` 阶段的保守语义特征候选。

该阶段当前是规则化 MVP，不调用 VLM，也不把结果当作最终 CAD 特征真值。它只根据 component bbox、面积、长宽比、视图 crop 尺寸和 `geometry_core` 质量信息，给出：

```text
outer_profile_candidate
hole_candidate
slot_candidate
annotation_residue_candidate
unknown_geometry_candidate
```

这些结果主要用于人工复核、后续尺寸-几何绑定、约束图构建和失败分析。进入论文或实验指标时，应把它定义为 `semantic feature candidate extraction`，而不是最终 `CAD feature recognition`。

批量处理所有已有 DrawingIR 样本，默认跳过 `-copy` 样本：

```bash
export PYTHONPATH=src

python -m vlm_cadcoder.cli extract-view-features \
  --dataflow-root DataFlow
```

单个样本：

```bash
python -m vlm_cadcoder.cli extract-view-features \
  --sample-id M001-08-006-B \
  --dataflow-root DataFlow
```

遇到坏 JSON 或缺失输入立即停止：

```bash
python -m vlm_cadcoder.cli extract-view-features \
  --dataflow-root DataFlow \
  --fail-fast
```

输出：

```text
DataFlow/08.Multi-viewFeatureExtraction/<sample_id>/view_features.json
DataFlow/08.Multi-viewFeatureExtraction/view_feature_summary.csv
DataFlow/08.Multi-viewFeatureExtraction/view_feature_summary.json
```

`view_features.json` 主要包含：

```text
views                     按 view_id 分组的语义候选、视图类型、bbox、crop_size 和 geometry_core 质量块
feature_candidates        扁平化语义候选列表，保留 source_candidate_id、bbox、bbox_on_page、metrics、evidence
skipped_components        DrawingIR 中无法匹配到正式 view 的低层 component
quality                   候选数量、跳过数量、是否需要人工复核和后续阻塞项
```

注意：

- 当前所有语义候选都带有 `needs_manual_review=true`；
- `hole_candidate`、`slot_candidate` 等名称只表示“候选”，不能直接用于 CAD 建模参数；
- 如果输出大量 `annotation_residue_candidate` 或 `unknown_geometry_candidate`，优先回查 `geometry_core.png` 质量和 `geometry_core_audit.json` 的 A/B/C 分层。

### 8.2 `10.StructuredCADRepresentation + dimension_ocr -> 08.DimensionCandidates`

用途：读取正式链路的 `drawing_ir.json` 和一个或多个 VLM/OCR `dimension_ocr` 预测文件，将尺寸文本归一化为可复核尺寸候选。该阶段只做候选抽取、基础类型推断、数量/数值/公差解析和 view 归属，不做尺寸-几何绑定。

输入：

```text
DataFlow/10.StructuredCADRepresentation/<sample_id>/drawing_ir.json
experiments/**/predictions.jsonl   # task=dimension_ocr
```

输出：

```text
DataFlow/08.Multi-viewFeatureExtraction/<sample_id>/dimension_candidates.json
DataFlow/08.Multi-viewFeatureExtraction/dimension_extraction_summary.csv
DataFlow/08.Multi-viewFeatureExtraction/dimension_extraction_summary.json
```

推荐先用单条命令扫描 DrawingIR 中所有正式非轴测视图。模型只加载一次，并将所有结果写入同一个 run dir：

```bash
export PYTHONPATH=src

python -m vlm_cadcoder.benchmarks.model_screening.runner \
  --model qwen2_5_vl_3b \
  --single-views \
  --dataflow-root DataFlow \
  --output-root experiments/dimension_ocr_single_views
```

只处理单个样本：

```bash
python -m vlm_cadcoder.benchmarks.model_screening.runner \
  --model qwen2_5_vl_3b \
  --single-views \
  --sample-id X476-07-011-C \
  --dataflow-root DataFlow \
  --output-root experiments/dimension_ocr_single_views
```

`--single-views` 默认执行 `dimension_ocr`，并且：

```text
只读取 DataFlow/10.StructuredCADRepresentation/*/drawing_ir.json 中的正式 views
使用每个 view 的 image_clean
默认跳过 type=isometric
为每条 prediction 写入 sample_id 和 view_id
一个运行目录只生成一个 predictions.jsonl
```

如需把轴测图也纳入输入，可显式传入 `--include-isometric`。

运行结束后，终端会打印真实 run dir，例如：

```text
experiments/dimension_ocr_single_views/20260620_153012_qwen2_5_vl_3b_dimension_ocr_single_views
```

再使用这个真实路径生成单个样本的尺寸候选：

```bash

python -m vlm_cadcoder.cli extract-dimensions \
  --sample-id X476-07-011-C \
  --dataflow-root DataFlow \
  --prediction-jsonl experiments/dimension_ocr_single_views/20260620_153012_qwen2_5_vl_3b_dimension_ocr_single_views/predictions.jsonl
```

对同一份 predictions 批量生成所有样本的尺寸候选：

```bash
python -m vlm_cadcoder.cli extract-dimensions \
  --dataflow-root DataFlow \
  --prediction-jsonl experiments/dimension_ocr_single_views/20260620_153012_qwen2_5_vl_3b_dimension_ocr_single_views/predictions.jsonl
```

可以重复传入多个预测文件：

```bash
python -m vlm_cadcoder.cli extract-dimensions \
  --dataflow-root DataFlow \
  --prediction-jsonl experiments/dimension_ocr/run_a/predictions.jsonl \
  --prediction-jsonl experiments/dimension_ocr/run_b/predictions.jsonl
```

`dimension_candidates.json` 主要包含：

```text
views                     按 view_id 分组的尺寸候选
dimension_candidates      扁平化尺寸候选，包含 text、normalized、dimension_type、bbox、bbox_on_page、value、quantity、tolerance、source
unmatched_records         无法匹配到 DrawingIR view 的 dimension_ocr 记录
quality                   候选数量、未匹配记录数量和后续阻塞项
```

说明：

- 当前 `bbox` 只有在 OCR/VLM 输出中提供时才会保留；没有 bbox 的尺寸候选仍会进入结果，但会带有 `missing_dimension_bbox` 复核原因；
- `dimension_type` 会对 `Φ/Ø/⌀`、`R`、`M`、`C`、角度和粗糙度做基础规则归一化；
- `value` 只表示文本中的主数值，例如 `4 x Φ 4.5` 会解析为 `quantity=4, value=4.5`；
- 输出仍是候选层，`ready_for_dimension_geometry_binding=false`。正式参数必须等下一步 `bind-dimensions-to-geometry` 完成。

### 8.3 `08.FeatureCandidates + 08.DimensionCandidates -> 09.DimensionGeometryBindings`

用途：读取 `view_features.json` 和 `dimension_candidates.json`，为尺寸候选生成可复核的几何绑定候选。该阶段以视觉模型判断为主，规则/CV 只负责生成 top-k 几何候选、距离/类型证据和冲突检查；输出仍是候选层，不是最终约束图。

先生成规则候选脚手架，便于本地检查输入是否齐全：

```bash
export PYTHONPATH=src

python -m vlm_cadcoder.cli bind-dimensions-to-geometry \
  --sample-id X476-07-011-C \
  --dataflow-root DataFlow
```

服务器上接入视觉模型进行绑定判断：

```bash
python -m vlm_cadcoder.cli bind-dimensions-to-geometry \
  --sample-id X476-07-011-C \
  --dataflow-root DataFlow \
  --model qwen2_5_vl_3b \
  --model-config configs/models.json
```

输出：

```text
DataFlow/09.Cross-viewGeometricReasoning/<sample_id>/dimension_geometry_bindings.json
DataFlow/09.Cross-viewGeometricReasoning/<sample_id>/overlays/<view_id>_binding_overlay.png
DataFlow/09.Cross-viewGeometricReasoning/dimension_geometry_binding_summary.csv
DataFlow/09.Cross-viewGeometricReasoning/dimension_geometry_binding_summary.json
```

`dimension_geometry_bindings.json` 主要包含：

```text
binding_candidates      规则生成的尺寸-几何绑定候选，供视觉模型和人工复核
vlm_requests            每个 view 给视觉模型的编号候选上下文、overlay 图路径和 prompt
vlm_responses           可选视觉模型输出
unbound_dimensions      规则候选为空的尺寸
ambiguous_bindings      规则 top 候选分数接近的歧义绑定
quality                 是否可进入 ConstraintGraph；MVP 默认仍需人工复核
```

若 clean view 图片存在且当前环境安装了 `Pillow`，该命令会额外生成绑定 overlay：尺寸候选标为 `D1/D2/...`，几何候选标为 `G1/G2/...`。服务器模型模式会优先把这张 overlay 图作为视觉输入；如果 overlay 无法生成，会自动退回 clean view，并在 `vlm_requests[].overlay_error` 中记录原因。
`vlm_binding_candidates` 会把模型返回的 `G*` 标签映射回真实几何 `feature_id`；若模型返回未出现在 `visual_labels.features` 里的 `G*`，该候选会标记 `invalid_candidate`，并写入 `vlm_target_label_not_in_request`。

说明：

- 规则候选只作为 `top-k target generation`，不等同于最终绑定；
- 若 `view_features.json` 为空，MVP 会退回使用 DrawingIR 中的低层 `geometry_component` 作为 `unknown_geometry_candidate`，仅用于给视觉模型提供编号候选；
- 对 `diameter/thread/chamfer/linear/angle` 做最小类型兼容过滤；
- `ready_for_constraint_graph=false`，正式 ConstraintGraph 需等待绑定候选经视觉模型/人工复核后再生成。

## 9. `06.SingleViews + experiments -> 10.StructuredCADRepresentation + 11 prompt`

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

## 10. `10.StructuredCADRepresentation -> 11.CADProgram` 规则草稿

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

## 11. `10/11 prompt -> 11.CADProgram` LLM 生成 CadQuery

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

## 12. CadQuery LLM 输出后处理

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

## 13. CadQuery 脚本执行

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

## 14. 建议的一次性运行顺序

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

正式样本从 `06.SingleViews` 到 `08.Multi-viewFeatureExtraction` 的当前推荐顺序：

```bash
export PYTHONPATH=src

python -m vlm_cadcoder.cli classify-views \
  --dataflow-root DataFlow

python -m vlm_cadcoder.cli generate-geometry-core-unet \
  --dataflow-root DataFlow \
  --sketchpic2viewpic-root /home/zxwcax/Projects/SketchPic2View/SketchPic2ViewPic \
  --python /home/zxwcax/anaconda3/envs/sketchpic2viewpic/bin/python \
  --skip-existing

python -m vlm_cadcoder.cli repair-geometry-core \
  --dataflow-root DataFlow \
  --skip-existing

python -m vlm_cadcoder.cli repair-geometry-primitives \
  --dataflow-root DataFlow \
  --skip-existing

python -m vlm_cadcoder.cli audit-geometry-core \
  --dataflow-root DataFlow

python -m vlm_cadcoder.cli build-drawing-ir \
  --dataflow-root DataFlow

python -m vlm_cadcoder.cli extract-view-features \
  --dataflow-root DataFlow
```

说明：

- `repair-geometry-core` 和 `repair-geometry-primitives` 当前都是辅助产物，不会自动替换 DrawingIR 的输入；
- 如果要做直线基元修复消融，在 `repair-geometry-primitives` 中显式加入 `--primitive-type line --primitive-type circle_arc`；
- 如果 `audit-geometry-core` 输出 B/C 或人工认为不可靠，应先更新 `geometry_core_audit_overrides.json`，再重新运行 `build-drawing-ir` 和 `extract-view-features`。

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

## 15. 后续需要补的命令

当前 `01 -> 10 -> 08` 的视图级 DrawingIR 与语义候选骨架已通过 LLM-CADCoder、SketchSegment 和 SketchPic2ViewPic 联动形成流程。下一步主线应从“几何候选”推进到“尺寸-几何绑定”和“可校验 CAD 生成”。建议后续补齐以下 CLI：

```text
extract-dimensions
bind-dimensions-to-geometry
build-constraint-graph
validate-cadquery-step
```

其中优先级最高的是：

```text
extract-dimensions
bind-dimensions-to-geometry
```

原因：当前 08 只提供保守语义特征候选，尚未把尺寸文本、尺寸线、箭头、引线和几何候选绑定起来。只有完成尺寸-几何绑定，CadQuery 生成才有足够的参数依据；`validate-cadquery-step` 则用于后续 CAD 执行闭环。

这些命令完成后，正式链路就可以从：

```text
01 -> 02 -> 03/04 -> 05 -> 06 -> 07
```

稳定推进到：

```text
06 -> 07 -> 08 -> 09 -> 10 -> 11
```
