# VLM4CAD：面向二维工程图纸的约束感知参数化 CAD 代码生成

## 1 课题定位

本项目面向智能制造场景中“二维工程图纸难以自动转化为可编辑三维 CAD 模型”的问题，研究一种基于多模态大模型、工程图纸结构化理解和尺寸约束推理的参数化 CAD 代码生成方法。

相比直接追求“图纸理解 -> CAD 建模 -> CAM 工艺 -> 报价”的完整工业链路，本课题将博士阶段主线收敛为：

```text
二维工程图纸
-> 图纸结构化理解
-> 尺寸-几何绑定与约束图构建
-> 参数化 CAD 建模脚本生成
-> 执行、渲染、校验与修复
-> 可编辑三维 CAD 模型
```

其中，CAM 工艺生成作为后期窄域扩展和工程应用验证，不作为当前主创新点。产品报价暂不纳入本阶段研究范围。

## 2 研究目标

本课题的核心目标是：从二维机械工程图纸中抽取视图、几何、尺寸、标注和约束知识，构建可用于参数化建模的中间表示，并生成可执行、可编辑、可校验的 FreeCAD 或 CadQuery 建模脚本，最终得到满足图纸尺寸与拓扑约束的三维 CAD 模型。

具体目标包括：

- 建立二维工程图纸到参数化 CAD 代码生成的任务定义和评测体系；
- 研究工程图纸中的视图、尺寸、符号、标注和加工注释的结构化理解方法；
- 构建尺寸-几何实体绑定关系和约束图，支撑参数化建模推理；
- 生成 FreeCAD/CadQuery 建模脚本，并通过执行反馈提高代码可用性；
- 通过 STEP 导出、多视角渲染和约束校验实现生成结果的自动评价与修复；
- 在典型机械零件族上验证方法有效性，并探索窄域 CAM 工艺生成的可行性。

## 3 输入与输出

### 3.1 输入数据

输入为二维机械工程图纸，优先从范围可控的典型零件族开始，例如法兰、轴套、板件、支架、箱体简化件等。

图纸可能包含：

- 主视图、俯视图、左视图、剖视图和局部放大视图；
- 轮廓线、中心线、隐藏线、剖面线等工程图线型；
- 长度、直径、半径、角度、公差等尺寸标注；
- 孔、槽、倒角、圆角、阵列、对称等几何特征；
- 材料、表面粗糙度、热处理、加工要求等工艺注释。

### 3.2 输出数据

主要输出包括：

- 结构化图纸知识表示；
- 尺寸-几何绑定关系；
- 几何约束图和参数依赖关系；
- FreeCAD 或 CadQuery 参数化建模脚本；
- STEP/BREP 等可交换三维模型；
- 生成结果的执行日志、尺寸误差、拓扑一致性和修复记录。

后期扩展输出：

- 面向孔加工、槽加工、轮廓铣削等窄域任务的 FreeCAD CAM/Path 工艺脚本。

## 4 技术路线

### 4.1 工程图纸结构化理解

该模块负责从二维工程图纸中抽取可建模信息。

重点任务包括：

- 图纸预处理与分辨率增强；
- 视图区域检测与切分；
- OCR 识别尺寸文本、符号和技术要求；
- 图线、箭头、尺寸界线、中心线等图形元素识别；
- 多视图之间的对应关系推断；
- 工程图纸知识结构化编码。

该阶段不直接生成 CAD 代码，而是输出稳定的图纸知识表示，降低端到端生成的不确定性。

### 4.2 尺寸-几何绑定与约束图构建

该模块是本课题区别于普通视觉识别和普通代码生成的关键。

工程图纸中的尺寸不是孤立文本，而是与几何实体、视图关系和建模参数共同构成约束系统。需要将尺寸标注绑定到对应的线、圆、孔、槽、边界、阵列或基准关系上，并建立约束图。

约束图需要表达：

- 几何实体及其参数；
- 尺寸标注与几何实体的关联；
- 对称、同轴、平行、垂直、阵列、相切等几何约束；
- 多视图之间的投影对应关系；
- 参数依赖、冗余约束和冲突约束。

### 4.3 约束感知的参数化 CAD 代码生成

该模块将结构化图纸知识和约束图转化为参数化建模脚本。

候选 CAD 后端：

- FreeCAD：适合工程建模、STEP 导出和后期 CAM/Path 扩展；
- CadQuery：适合 Python 风格参数化建模、代码可读性和几何验证。

建模脚本生成不应仅依赖 VLM 直接输出代码，而应采用“结构化知识 -> 建模规划 -> 代码生成”的分层策略：

```text
图纸知识
-> 建模特征序列
-> 参数表
-> 草图/拉伸/切除/倒角/阵列等操作
-> CAD 脚本
```

直接 VLM-to-FreeCAD 代码生成可作为 baseline，用于证明约束感知方法的改进效果。

### 4.4 执行反馈、几何校验与代码修复

生成脚本必须通过 CAD 内核执行验证。

校验流程包括：

- 执行 CAD 脚本，检查语法错误、API 错误和建模失败；
- 导出 STEP/BREP 模型；
- 渲染多视角图像，与原工程图视图进行对比；
- 检查尺寸误差、拓扑结构、孔槽数量和特征位置；
- 根据错误类型进行定向修复。

错误类型可划分为：

- 语法错误；
- CAD API 调用错误；
- 参数缺失或单位错误；
- 几何尺寸错误；
- 拓扑结构错误；
- 多视图约束不一致；
- 代码可执行但模型不可用。

### 4.5 窄域 CAM 工艺生成扩展

CAM 生成不作为当前主线，而作为 CAD 模型生成后的工程应用验证。

建议优先选择范围可控的 2.5D 加工场景，例如：

- 孔加工；
- 槽加工；
- 外轮廓铣削；
- 平面加工。

该模块需要依赖已生成的 CAD 特征、材料信息、加工注释和工艺规则，探索 FreeCAD CAM/Path 脚本生成的可行性。

## 5 创新点提炼

当前建议将博士阶段创新点凝练为以下方向：

1. 面向工程图纸的结构化知识抽取方法；
2. 尺寸-几何绑定与约束感知中间表示；
3. 基于约束图的参数化 CAD 建模脚本生成方法；
4. 执行反馈驱动的 CAD 代码校验与修复机制；
5. 面向二维图纸到 CAD 代码生成任务的数据集、指标与评测体系；
6. 窄域 CAM 工艺生成的工程扩展验证。

## 6 评测指标

为避免课题停留在系统演示层面，需要建立可量化评价体系。

建议指标包括：

- 视图检测准确率；
- OCR 文本识别准确率；
- 尺寸标注识别准确率；
- 尺寸-几何实体绑定准确率或 F1 值；
- 约束图一致性；
- CAD 脚本可执行率；
- STEP 导出成功率；
- 关键尺寸误差；
- 特征数量匹配率；
- 拓扑一致性；
- 渲染视图相似度；
- 修复成功率和平均修复轮次。

## 7 实验设计

实验应至少包括：

- 与直接 VLM-to-CAD 代码生成方法的对比；
- 与文本描述到 CAD 代码生成方法的对比；
- 无约束图、无尺寸绑定、无执行修复等消融实验；
- 不同图纸质量、分辨率和标注复杂度下的鲁棒性实验；
- 不同零件族上的泛化实验；
- 典型失败案例分析。

## 8 当前原型

当前代码仍处于早期验证阶段：

- `test_internvl.py`：测试 InternVL 对单张图纸的 FreeCAD 代码生成能力；
- `Sketch2CAD.py`：调用 InternVL 生成 FreeCAD 脚本，并尝试通过 FreeCAD 导出 STEP；
- `generated_freecad.py`：简单 FreeCAD 建模脚本示例。

当前已新增第一版工程骨架：

- `configs/`：模型、benchmark 任务和 DataFlow 配置；
- `data/`：样本索引、benchmark split 和人工标注目录；
- `src/vlm_cadcoder/dataflow/`：PDF 渲染、样本索引和阶段产物管理；
- `src/vlm_cadcoder/models/`：可插拔 VLM/OCR 模型适配器接口；
- `src/vlm_cadcoder/benchmarks/model_screening/`：小模型筛选 benchmark 任务、prompt、runner 和评测函数；
- `src/vlm_cadcoder/ir/`：DrawingIR 数据结构；
- `src/vlm_cadcoder/cad/`：后续 CadQuery 代码生成与几何校验模块入口。

后续需要补充真实实现：

- 数据集与标注格式；
- 图纸结构化解析模块；
- 约束图表示与校验模块；
- CAD 代码生成模块；
- 执行、渲染、评测与修复模块；
- 实验脚本和结果记录。

## 9 项目进度与当前阶段

截至 2026-06-11，项目已完成 `01.RawPDFWithSTEP -> 06.SingleViews` 的第一版数据流闭环。当前重点不是直接追求完整 CAD 自动生成，而是先把图纸理解输入质量、页面清理、视图检测、单视图裁剪和小模型筛选 benchmark 固化成可复现、可评测的工程基础，然后进入视图分类、特征抽取、尺寸-几何绑定和 DrawingIR 构建。

### 9.1 已完成事项

已完成的课题层面工作：

- 将课题主线从“图纸理解 -> CAD -> CAM -> 报价”的完整工业链路收敛为“面向二维工程图纸的约束感知参数化 CAD 代码生成”；
- 明确 CAM 工艺生成作为后期窄域扩展，不作为当前主创新点；
- 明确 CadQuery 更适合作为主要 CAD 代码生成目标，FreeCAD 更适合作为后期工程验证、可视化和 CAM/Path 扩展平台；
- 明确图纸理解阶段的核心任务包括视图切分、OCR、尺寸-几何绑定和约束图构建；
- 确定第一批数据应优先选取板类、法兰类、轴套类、主基体拉伸加标准减材特征的棱柱类零件。

已完成的数据与工程准备：

- 建立 `DataFlow/` 阶段目录，覆盖从原始 PDF/STEP 到 CAD 程序的 11 个阶段；
- 在 `DataFlow/01.RawPDFWithSTEP/` 中初步收集 PDF 图纸及 STEP 真值；
- 增加 `DataFlow/README.md`，定义各阶段产物和图像坐标规范；
- 保留早期原型脚本 `test_internvl.py`、`Sketch2CAD.py`、`PDF2PNG.py` 和简单 FreeCAD/CadQuery 示例。
- 实现 `01 -> 02` PDF 渲染、`02 -> 03/04` 页面级 layout 分析与 clean PNG 生成；
- 初步实现 `clean-layout` 页面级清理流程，可识别并移除外边框、内图框条带、角落元信息框、标题栏、左侧孔表、右上角版本表等非视图区区域，同时将有语义价值的移除区域分类保存为 crop，供后续语义抽取使用；
- 接入 SketchSegment 作为 `04 -> 05` 的视图候选检测工具，并在 LLM-CADCoder 中加入 `ViewCandidateFilter` 后置过滤；
- 通过 SketchSegment 导出脚本完成 `05 -> 06` 单视图裁剪，当前 `DataFlow/06.SingleViews/` 已形成一批自动裁剪样本；
- 实现 `audit-single-views` 和 `classify-views`，可对 `05/06` 一致性进行审计，并生成 `07.ViewClassification` 的启发式视图类型 baseline；
- 实现 `build-drawing-ir`，可从 `05.ViewDetection`、`06.SingleViews` 和 `07.ViewClassification` 生成正式链路的 DrawingIR v0.1 视图级骨架；
- 已在 `DataFlow/LayoutSamples/`、`DataFlow/03.LayoutAnalysis/`、`DataFlow/04.CleanPNG/`、`DataFlow/05.ViewDetection/` 和 `DataFlow/06.SingleViews/` 上进行若干样例验证，当前清理与裁剪策略以“保留视图及其相关标注、去除表格/边框/页面元信息”为目标。

已完成的 benchmark 工程骨架：

- 新增 `configs/`，用于管理模型、benchmark 任务和 DataFlow 配置；
- 新增 `data/samples.csv` 和 `data/splits/benchmark_small.jsonl`，作为样本索引与小规模 benchmark split 的起点；
- 新增 `src/vlm_cadcoder/dataflow/`，用于样本索引、PDF 渲染和阶段产物路径管理；
- 新增 `src/vlm_cadcoder/models/`，提供统一 VLM/OCR 模型适配器接口；
- 新增 `src/vlm_cadcoder/benchmarks/model_screening/`，提供小模型筛选任务、prompt、runner 和基础评测函数；
- 新增 `src/vlm_cadcoder/ir/`，提供 DrawingIR 的初始数据结构；
- 新增 `envs/`，为服务器端 conda 环境配置提供基础文件。

### 9.2 当前所处阶段

当前处于：

```text
第二阶段：图纸理解和中间表示抽取
当前子阶段：01-07 数据流验收与 DrawingIR v0.1 视图级骨架构建
```

当前阶段的核心目标是：

- 建立稳定的 PDF 图纸处理流程；
- 建立稳定的页面级 layout cleaning 流程，降低整张 A4/A3 图纸中边框、标题栏和表格对 VLM 的干扰；
- 建立自动 `05.ViewDetection` 与 `06.SingleViews` 视图裁剪流程，并对其进行人工验收和定量评估；
- 允许使用外部方法切好的 `DataFlow/06.SingleViews/testView2CAD/` crops 作为临时原型输入，提前验证后续 DrawingIR、特征抽取和 CadQuery 代码生成流程；
- 形成可复现的小模型筛选 benchmark；
- 比较 InternVL、Qwen-VL、PaddleOCR-VL 等候选模型在工程图纸理解任务上的表现；
- 明确后续图纸理解主模型和 OCR/layout 工具组合；
- 为 08 特征抽取、尺寸-几何绑定和约束图构建提供可靠输入。

当前尚未进入：

- 大规模模型训练或微调；
- 完整 DrawingIR 自动生成，当前仅完成视图级骨架；
- 尺寸-几何绑定算法定型；
- ConstraintGraph 自动构建；
- 正式的 CadQuery 脚本生成模块；
- CAD 执行反馈修复；
- CAM 工艺生成。

但可以使用人工或外部工具切好的单视图 crops 先做下游原型闭环。该闭环只用于验证“单视图/多视图输入是否足以抽取建模参数并生成目标 CadQuery 代码”，不作为最终方法效果评价。

### 9.3 近期优先任务

近期优先任务按依赖关系排列：

1. 冻结一版 `01 -> 06` 数据流版本，记录 layout cleaner、SketchSegment 权重、view filter 和导出脚本版本；
2. 对 `06.SingleViews` 做人工验收表，标记 expected view count、actual view count、false positive、missed view、crop quality 和备注；
3. 统一 `05.ViewDetection` 与 `06.SingleViews` 的来源关系，确保 `06` 只由过滤后的 `05/page_001_views.json` 导出；
4. 将 `copy`、旧版本或未过滤样本隔离为失败分析/消融样本，避免污染正式评测；
5. 人工复核 `07.ViewClassification` 中低置信度或 `needs_manual_review=true` 的视图类型，形成第一版主视图/侧视图/俯视图/轴测图标签；
6. 基于 `drawing_ir.json` 设计 `08.Multi-viewFeatureExtraction` 的最小输出 schema，先覆盖板类件中的孔、槽、倒角、圆角、阵列和厚度/拉伸方向；
7. 跑通 `view_count`、`view_classification`、`dimension_ocr`、`feature_count`、`json_stability` 五个小模型筛选任务，并比较 full page、clean page、single-view crop 三种输入；
8. 基于评测结果确定主 VLM、OCR 工具、layout/view crop 工具组合和后续 DrawingIR 扩展方式。

### 9.4 后续阶段计划

后续阶段按研究主线推进：

| 阶段 | 目标 | 主要产物 | 状态 |
| --- | --- | --- | --- |
| 阶段 1：课题收敛与数据准备 | 明确博士主线、样本范围和数据流 | README、DataFlow、初始 PDF/STEP 数据 | 已完成第一版，持续补充 |
| 阶段 2：图纸理解与模型筛选 | 评估小参数 VLM/OCR 模型的图纸理解能力，建立 layout/view crop 数据流 | model-screening benchmark、layout cleaning、view crop、评测报告 | `01 -> 07` 已形成基线，当前验收中 |
| 阶段 3：DrawingIR 构建 | 从 page/view crops 抽取视图、尺寸、标注和特征候选 | DrawingIR JSON、标注规范 | v0.1 视图级骨架已实现，下一步补 08 特征抽取 |
| 阶段 4：尺寸-几何绑定与约束图 | 建立尺寸标注、几何实体和多视图关系 | ConstraintGraph、绑定算法、F1 指标 | 待完成 |
| 阶段 5：CadQuery 代码生成 | 将结构化图纸知识转为参数化建模脚本 | CadQuery 代码、STEP 模型 | 可用外部 crops 做临时原型 |
| 阶段 6：执行反馈与修复 | 根据 CAD 执行、渲染和尺寸误差修复代码 | 修复日志、几何校验报告 | 待完成 |
| 阶段 7：窄域 CAM 扩展 | 针对孔加工、槽加工等任务做工程验证 | FreeCAD CAM/Path 脚本 | 长期扩展 |

### 9.5 当前风险与约束

当前主要风险包括：

- PDF 图纸渲染质量会直接影响 OCR、视图切分和尺寸绑定结果；
- layout cleaning 若误删视图线或局部标注，会直接影响后续 DrawingIR 和 CAD 参数生成，因此必须保留 overlay、mask 和 removed-region crops 便于追溯；
- 外部工具切好的 `testView2CAD` crops 可以用于后半段原型验证，但必须记录其来源，不能作为自动 view detection 方法的正式评测结果；
- VLM 对工程图纸的全局理解能力不等于尺寸-几何绑定能力，必须单独评测；
- 仅靠 VLM 直接输出 CAD 代码不稳定，应作为 baseline，而不是主方法；
- 如果样本零件过早扩展到复杂曲面、装配体或多方向复杂加工，研究范围会失控；
- PaddleOCR-VL 与 PyTorch VLM 环境可能存在依赖冲突，建议在服务器端独立服务化。

## 10 小模型筛选 Benchmark

第一阶段 benchmark 用于筛选适合工程图纸理解的小参数 VLM，而不是直接评价 CAD 生成。

候选任务包括：

- `view_count`：视图数量识别；
- `view_classification`：主视图、俯视图、左视图、剖视图等视图类型判断；
- `dimension_ocr`：尺寸文本、符号和标准化尺寸语义识别；
- `feature_count`：孔、槽、沉孔、倒角、圆角等 CAD 特征数量统计；
- `json_stability`：结构化 JSON 输出稳定性。

模型通过 `configs/models.json` 切换，当前预置：

- `mock`；
- `internvl_2b`；
- `internvl3_5_2b`；
- `qwen3_vl_2b_instruct`；
- `qwen3_vl_2b_thinking`；
- `qwen2_5_vl_3b`；
- `paddleocr_vl`。

示例命令：

```bash
python -m vlm_cadcoder.cli build-sample-index \
  --raw-dir DataFlow/01.RawPDFWithSTEP \
  --output data/samples.csv

python -m vlm_cadcoder.cli render-pdf \
  --pdf DataFlow/01.RawPDFWithSTEP/X350-05-070-A.pdf \
  --sample-id X350-05-070-A \
  --dpi 600 \
  --skip-multipage

python -m vlm_cadcoder.benchmarks.model_screening.runner \
  --model mock \
  --task view_count \
  --image DataFlow/02.RawPNG/X350-05-070-A/page_001_600dpi.png
```

真实模型集成时，只需要实现对应 adapter：

- `src/vlm_cadcoder/models/internvl_adapter.py`；
- `src/vlm_cadcoder/models/qwen_vl_adapter.py`；
- `src/vlm_cadcoder/models/paddleocr_vl_adapter.py`。

当前工程约定：

- 本地开发阶段只维护代码、配置和数据流结构；
- 不在本地安装 VLM/CUDA/PaddleOCR 环境；
- 不在本地运行模型或 benchmark；
- 服务器端根据 `envs/` 中的 conda 环境文件配置运行环境。

服务器环境文件：

- `envs/vlm-cadcoder-server.yml`：PyTorch、Transformers、Qwen/InternVL、PDF 渲染主环境；
- `envs/paddleocr-vl-server.yml`：PaddleOCR-VL 独立环境，建议作为 OCR/layout 服务接入。

PaddleOCR-VL 建议独立服务化，避免 Paddle 与 PyTorch VLM 环境发生依赖冲突；主 benchmark 通过 `configs/models.json` 中的 `server_url` 调用。

## 11 View2CAD 外部裁剪原型

`DataFlow/06.SingleViews/testView2CAD/` 中的外部裁剪视图可用于提前验证下游 View2CAD 闭环。该流程会汇总外部 crops、clean 图、VLM benchmark 输出和 `DataFlow/01.RawPDFWithSTEP/testView2CAD/` 中的 STEP 真值，生成最小 DrawingIR、建模计划和 CadQuery 生成提示。

示例命令：

```bash
python -m vlm_cadcoder.cli build-view2cad-prototype \
  --sample-id 2023-2024-1-923 \
  --dataflow-root DataFlow \
  --experiments-root experiments/external_crops
```

输出产物：

```text
DataFlow/10.StructuredCADRepresentation/testView2CAD/<sample_id>/external_crop_manifest.json
DataFlow/10.StructuredCADRepresentation/testView2CAD/<sample_id>/minimal_drawing_ir.json
DataFlow/10.StructuredCADRepresentation/testView2CAD/<sample_id>/modeling_plan.json
DataFlow/11.CADProgram/testView2CAD/<sample_id>/cadquery_generation_prompt.md
```

在人工复核前，可继续生成 CadQuery 参数表和草稿脚本：

```bash
python -m vlm_cadcoder.cli build-cadquery-draft \
  --sample-id 2023-2024-1-923 \
  --dataflow-root DataFlow
```

输出产物：

```text
DataFlow/11.CADProgram/testView2CAD/<sample_id>/cadquery_parameters.json
DataFlow/11.CADProgram/testView2CAD/<sample_id>/cadquery_draft.py
```

也可以让服务器上的 VLM/LLM 基于 `cadquery_generation_prompt.md`、clean 图和 external crops 直接生成 CadQuery 代码：

```bash
python -m vlm_cadcoder.cli generate-cadquery-llm \
  --sample-id 2023-2024-1-923 \
  --model qwen2_5_vl_3b \
  --dataflow-root DataFlow \
  --max-new-tokens 4096
```

该命令会保存原始模型输出，并自动剥离 markdown fence、规范化 CadQuery import 和 STEP export：

```text
DataFlow/11.CADProgram/testView2CAD/<sample_id>/cadquery_llm_generated.raw.md
DataFlow/11.CADProgram/testView2CAD/<sample_id>/cadquery_llm_generated.py
```

如果已有模型输出包含 markdown fence 或错误 import，可只做后处理：

```bash
python -m vlm_cadcoder.cli sanitize-cadquery-llm \
  --input DataFlow/11.CADProgram/testView2CAD/2023-2024-1-923/cadquery_llm_generated.py
```

该流程属于原型闭环：可以验证单视图/多视图 crops 是否足以支撑 CadQuery 代码生成，但不能替代正式的自动视图检测、尺寸-几何绑定和约束图评测。

## 12 软硬件环境

计划环境：

- 系统：Ubuntu 22.04；
- GPU：RTX 4090；
- 开发工具：PyCharm、VS Code；
- 视觉模型：InternVL 系列或其他多模态大模型；
- CAD 后端：FreeCAD、CadQuery；
- 输出格式：FreeCAD Python、CadQuery Python、STEP/BREP。

## 13 研究边界

本阶段不直接承诺解决以下问题：

- 任意工业图纸到完整三维 CAD 的通用生成；
- 完整 CAM 工艺规划；
- 报价和成本估算；
- 复杂装配体与运动机构生成；
- 真实工厂级工艺知识闭环。

这些内容可作为长期愿景或后续扩展，但当前博士主线应优先保证“二维工程图纸到约束感知参数化 CAD 代码生成”这一核心问题能够被深入研究、系统验证和量化评价。
