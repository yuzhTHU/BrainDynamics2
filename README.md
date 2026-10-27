# BrainDynamicsCode

本仓库用于从脑神经观测数据中建立动力学方程和网络模型，并据此研究群体活动、状态转换、空间传播、有效连接及模型可辨识性。

## 目录约定

```text
BrainDynamicsCode/
├── README.md
├── data/
│   ├── README.md
│   ├── raw/
│   └── processed/
├── paper/
│   └── *.pdf
├── scripts/
│   ├── *.ipynb
│   └── *.py
├── figures/
│   └── <scripts内文件名>/
│       ├── *.png
│       ├── *.svg
│       └── *.pdf
├── logs/
│   └── <scripts内文件名>/
│       └── *
├── notes/
│   └── *.md
└── venv/
```

各目录的职责如下。

### `data/`

保存原始数据与整理后的建模输入。具体文件、数据形状、同步关系和解释限制见 [`data/README.md`](data/README.md)。

- `data/raw/` 中的原始数据只读，不得就地修改。
- `data/processed/` 保存从原始数据整理出的通用输入，也应尽量保持稳定。
- 当前数据文件是指向既有文件的硬链接；修改它们等同于修改源文件，请避免修改。
- 新的派生数据不得覆盖现有文件。若确有必要，应使用新文件名，并在相应笔记或脚本中记录生成方法。

### `paper/*.pdf`

保存研究过程中实际阅读或引用的论文 PDF，尤其是难以稳定在线访问、需要认证或会被后续实验直接引用的文献。

- 文件名应尽量包含第一作者、年份和简短题目，例如 `Muller2018_CorticalTravellingWaves.pdf`。
- 文献结论、适用范围及其与本项目的关系写入 `notes/`，不要只保存 PDF 而不留研究记录。
- 不保存与当前研究无关的大批量文献。

### `scripts/*.{py,ipynb}`

保存所有可执行的分析与实验代码。

- `*.ipynb`：用于快速数据分析、交互式探索、结果核查和可视化。
- `*.py`：用于运行时间较长、需要重复执行或适合批处理的任务，例如神经网络训练、大规模参数扫描、bootstrap、模型比较和数据预处理。
- notebook 与脚本都应优先使用相对于仓库根目录的路径，避免写死个人机器上的绝对路径。
- 可复用计算应逐步从 notebook 提取到 Python 脚本或函数中；notebook 主要负责组织分析、解释结果和展示少量关键输出。
- 长时任务必须把结果写入 `logs/` 或 `figures/`，不要只保存在终端输出或 notebook 临时状态中。

### `figures/*/*.{png,svg,pdf}`

保存图片格式的运行结果。一级子目录应与产生这些图片的分析名称一致，例如：

```text
scripts/261027空间事件传播.ipynb -> figures/261027空间事件传播/
```

- 图片文件使用能描述内容的名称，不使用 `figure1_final_v2` 一类难以追踪的名称。
- 推荐同时保存 SVG、PDF 和 PNG；其中 SVG/PDF 用于论文和进一步编辑，PNG 用于快速浏览。
- 统一绘图规范见本文的“分析与绘图规范”一节。

### `logs/*/`

保存图片以外的运行结果，例如：

- 终端日志、训练曲线原始数值和配置快照；
- CSV、JSON、NPY、HDF5 等中间结果或汇总表；
- 缓存、checkpoint、模型权重和预测结果；
- 参数扫描、重复实验及随机种子对应的输出。

推荐按“任务/实验”组织：

```text
logs/
└── <任务名称>/
    └── <实验名称>/
        ├── config.json
        ├── metrics.csv
        ├── run.log
        └── checkpoints/
```

例如：

```text
scripts/261027_train_neural_ode.py -> figures/261027_train_neural_ode/seed42/
```

同一实验应记录随机种子、输入版本、关键参数、开始时间和代码版本，使结果可以复现。

此目录不纳入版本控制。

### `notes/*.md`

保存研究计划、数据审计、文献整理、实验结论、失败尝试、方法判断和后续问题。

- 笔记应说明证据来自数据、文献、模型结果还是推断。
- 重要结论应链接到对应脚本、notebook、图片或日志目录。
- 不把只有命令输出而没有解释的文件当作研究笔记。

### `venv/`

保存使用 `conda create -p ./venv python=3.12 -y` 创建的项目环境，不纳入版本控制。


## 命名约定

### 笔记与 notebook

笔记和 notebook 使用：

```text
YYMMDD中文说明.<扩展名>
```

例如：

```text
notes/261027空间方向反证实验.md
scripts/261027空间事件传播.ipynb
figures/261027空间事件传播/
```

- 日期采用六位 `YYMMDD`，表示文件首次形成的日期。
- 日期之后直接使用简洁的中文内容描述。
- 同一个分析的 notebook、图片目录和必要时的日志目录尽量使用相同名称。
- 同一天出现同类分析时，用语义明确的后缀区分，不使用“新版”“最终版2”等名称。

### Python 脚本与实验目录

Python 脚本可使用简洁的英文以便命令行调用，例如 `261027_train_neural_ode.py`。其日志目录应采用相同的任务名，并在下一层区分实验配置。

## 路径和结果管理

- 代码默认从仓库根目录运行，所有输入与输出路径均相对于仓库根目录解释。
- 图片只写入 `figures/`，其他运行产物只写入 `logs/`。
- 原始数据不得修改；任何变换都必须由脚本生成并可追溯。
- 不依赖 notebook 的隐式执行顺序。提交或归档关键 notebook 前，应从干净内核按顺序执行一次。
- 结论不能只存在于图中；关键发现、负结果和限制应同步写入 `notes/`。

## 分析与绘图规范

本节规定分析图的推荐样式。notebook 和 Python 分析脚本放在 `scripts/`；生成的图片写入与分析同名的 `figures/<分析名称>/` 目录。

### 基本规格

- 主绘图区高度优先使用 **6 cm** 或 **4 cm**。
- 主绘图区高宽比优先使用 **1:1**、**1:1.414** 或 **1:2**（这里的比例顺序是“高:宽”）。
- 正文、坐标轴标题和需要强调的文字使用 **14 pt**；刻度、图例和辅助标注可使用 **10 pt**，尽量不要使用其它字号。
- 字体使用 **Arial**。
- 线宽使用 **1.0 pt**，必要时可引入少量 1.5pt 的粗线条（例如需要强调的主要结果）和 0.5pt 的细线条（例如 grid），但应尽量避免。
- 分辨率使用 **300 dpi**。
- 保存格式为 **SVG、PDF、PNG**。
- 保存时使用透明背景。
- 复杂绘图元素应栅格化；例如散点数 `n > 20` 时设置 `rasterized=True`。坐标轴、文字和简单曲线仍保留为矢量元素。

推荐尺寸组合如下：

| 用途 | 高度 | 高:宽 | 宽度 |
|---|---:|---:|---:|
| 紧凑单图 | 4 cm | 1:1 | 4 cm |
| 常规单图 | 6 cm | 1:1 | 6 cm |
| 横向单图 | 6 cm | 1:1.414 | 8.484 cm |
| 宽图或时间序列 | 6 cm | 1:2 | 12 cm |

若有排版限制，可以在上述组合中选择最接近的一种；不要随意产生大量彼此只差少量尺寸的变体。

### 颜色顺序

颜色数量增加时，严格按下列顺序加入：

| 顺序 | 颜色 | 十六进制值 |
|---:|---|---|
| 1 | 蓝色 | `#4a6fe3` |
| 2 | 红色 | `#d33f6a` |
| 3 | 紫色 | `#bb65b6` |
| 4 | 绿色 | `#71a451` |
| 5 | 青色 | `#5f99ce` |
| 6 | 黄色 | `#c57b3e` |

也即：

- 只涉及一个颜色时，使用蓝色 `#4a6fe3`；
- 涉及两个颜色时，使用蓝色和红色；
- 涉及三个颜色时，再加入紫色；
- 需要更多颜色时，依次加入绿色、青色和黄色。

颜色应在同一项目中保持语义稳定。例如，一旦蓝色表示真实数据、红色表示模型结果，后续同类图中不应交换两者。

### 推荐绘图模板

项目优先使用 `yumeow_plot.get_fig`。下面给出一个 6 cm 高、1:1.414 的单图模板；`AW`、`AH` 分别表示主绘图区的宽和高，边距使用 5。

```python
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from yumeow_plot import get_fig

COLORS = [
    "#4a6fe3",  # 蓝
    "#d33f6a",  # 红
    "#bb65b6",  # 紫
    "#71a451",  # 绿
    "#5f99ce",  # 青
    "#c57b3e",  # 黄
]

mpl.rcParams.update({
    "font.family": "Arial",
    "font.size": 14,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "lines.linewidth": 1.0,
    "axes.linewidth": 1.0,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.transparent": True,
    "svg.fonttype": "none",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

height_cm = 6
width_cm = 1.414 * height_cm

fig, axes = get_fig(
    RN=1, CN=1,
    AW=width_cm, AH=height_cm,
    LM=5, RM=5, TM=5, BM=5,
    VS=5, HS=5,
    fontsize=14, dpi=300, lw=1,
)
ax_loader = iter(axes)
ax = next(ax_loader)

# n > 20 的散点等复杂元素使用 rasterized=True。
ax.scatter(x, y, s=12, color=COLORS[0], rasterized=(len(x) > 20))
ax.set_xlabel("横轴")
ax.set_ylabel("纵轴")

output_dir = Path("figures/261028示例分析")
output_dir.mkdir(parents=True, exist_ok=True)
for suffix in ("svg", "pdf", "png"):
    fig.savefig(
        output_dir / f"示例图.{suffix}",
        dpi=300,
        transparent=True,
        bbox_inches="tight",
    )
plt.close(fig)
```

如果当前环境尚未安装 `yumeow_plot`，可以暂时使用 Matplotlib 的 `figsize=(width_cm / 2.54, height_cm / 2.54)` 创建图，但最终图仍应满足本节中的尺寸、字体、线宽、颜色、透明背景和保存格式约定。

### 栅格化原则

- `scatter` 的点数 `n > 20` 时使用 `rasterized=True`。
- 大型热图、密集网格、置信区间集合或包含大量 patch 的对象，可根据文件大小和渲染速度决定是否栅格化。
- 文字、坐标轴、图例、稀疏曲线和少量标记尽量保持矢量格式。
- 栅格化只影响复杂绘图元素，不应把整张 SVG/PDF 退化成单一位图。

### 保存和命名

每个分析使用独立图片目录，目录名与脚本或 notebook 的文件名主体一致：

```text
scripts/261027区域方程库比较.ipynb
figures/261027区域方程库比较/
├── 模型误差比较.svg
├── 模型误差比较.pdf
└── 模型误差比较.png
```

- 图名描述图中比较的对象或结论，不使用无语义编号。
- SVG、PDF、PNG 三种格式使用相同的文件名主体。
- 临时调试图也应放入对应分析目录；确认无用后再清理，避免散落在仓库根目录或 `scripts/` 中。
- 图对应的关键结论和解释写入 `notes/`，不能只依赖图片文件本身。
