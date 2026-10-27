# Brain dynamics dataset

本目录汇集同一只小鼠、一次连续记录的原始文件与整理后的建模输入。数据文件均为指向既有文件的**硬链接**，不是额外副本；不要就地修改。如需改变数据，应另写派生文件。

## 目录

```text
data/
├── raw/
│   ├── wholebrain_output_after_rejection.mat
│   ├── valid_area.mat
│   ├── stimuli.mat
│   ├── stimuli.csv
│   ├── 51.mp4
│   └── readme.txt
├── processed/
│   ├── data.npy
│   ├── node_info.csv
│   └── time_info.csv
└── README.md
```

## 处理后数据

### `processed/data.npy`

- NumPy float32 数组，形状 `(28801, 62130)`：时间 × 节点。
- 它是原始 `wholebrain_output_after_rejection.mat:/whole_trace_ori` 的直接数值副本；随机抽查 10,000 个元素逐位相等。
- 转换时没有再做标准化、重采样或数值变换。
- 原始说明将 `whole_trace_ori` 称为“发放率信息”，但没有单位或预处理定义。数值约跨越 `-39,556` 到 `403,302`，因此建议使用中性名称 **activity trace**，不要直接解释成 Hz、膜电位或已校准事件率。

### `processed/node_info.csv`

- 62,130 行，与 `data.npy` 的列一一对应。
- 包含脑区缩写、脑区名称、atlas ID 和二维投影坐标。
- 与 `raw/valid_area.mat` 中的 `acs`、`names`、`ids`、`whole_center_2d_T` 完全一致。
- 共 25 个脑区标签；名称均为 layer 1。
- 没有 E/I、转录型或其他细胞类型标签。

### `processed/time_info.csv`

- 28,801 行，与 `data.npy` 的时间轴一一对应。
- 状态标签为 `Awake`、`NREM`、`REM`，由头皮 EEG 给出。
- 状态计数：Awake 17,105 帧，NREM 9,622 帧，REM 2,074 帧。
- 共 98 个连续 episode；REM 只有 5 段。
- 它与 `raw/stimuli.mat` 的 98 个 MATLAB 1-based 闭区间完全一致，并无缝覆盖全部帧。

## 原始文件

### `raw/readme.txt`

吴嘉敏老师课题组同学提供的原始说明。

### `raw/wholebrain_output_after_rejection.mat`

HDF5/MAT 文件。主要数据：

- `whole_trace_ori`：磁盘形状 `(28801, 62130)`，float32，对应处理后的 `data.npy`。
- `whole_center`：`(4, 62130)` 整数数组。前三行很可能是原始三维坐标；范围分别为 5–3274、2–4472、10–53。第四行范围 3–12，语义未知，不应擅自命名。

### `raw/valid_area.mat`

包含二维投影坐标、脑区名称、缩写和 atlas ID，以及 atlas/CCF 相关数据。处理后的 `node_info.csv` 来源于此文件。

### `raw/stimuli.mat` 与 `raw/stimuli.csv`

包含 Awake/NREM/REM 的连续区间。CSV 开头额外有一个 `0`；整理后的 `time_info.csv` 丢弃该占位值后与 MAT 区间一致。

### `raw/51.mp4`

- H.264，640×480，约 30 fps，时长约 8,376.13 s。
- 画面为固定头小鼠面部，可提取面部运动、眼部变化和胡须运动等协变量。
- 视频顶部绿光形成一段连续的 28,801 个脉冲，恰好等于神经数据帧数，是很强的逐帧同步证据。

候选同步关系：

```text
t_video_seconds ≈ 13.58266996 + 0.2411780956 * neural_frame
sampling_rate ≈ 4.146313525 Hz
```

- 神经记录有效跨度约 6,945.93 s。
- 一帧约 0.24118 s。
- 线性时钟拟合残差标准差约 26 ms，主要受 30 fps 视频时间量化影响。
- 绿灯高电平宽度中位数约 0.10 s；它只能作为采集/积分窗口的候选代理，不能当作正式硬件曝光参数。
- 精确对齐应使用每个脉冲的实际时间，而不是只使用一个固定 offset。

## 已知的数据结构

- 1024 个分层抽样节点中，PC1/前 5/前 20 个主成分约解释 36.4%/53.6%/63.8% 方差，存在很强的共同低维结构。
- raw trace 在几十秒尺度仍有较强相关，但逐 episode 去均值或线性去趋势后，约 8 s 起相关已接近零；长记忆主要来自 episode 基线和慢漂移。
- 4.146 Hz 采样意味着亚帧时延不可辨识。一个离散 lag 不能直接解释为突触时延。
- 区域平均事件在同一帧附近达峰；更细的空间 bin 中可检测弱而可复现的时空梯度，但时间反转同样可预测，因此不能仅凭这些梯度宣称有效连接或物理传播方向。

## 适合与不适合的问题

较适合：

- 秒级及更慢的群体动力学；
- Awake/NREM/REM 条件动力学与状态转换；
- 低维共同模态、区域耦合、空间扩散/平流候选；
- 长期统计、稳定性、群体事件及行为协变量控制。

不适合单独回答：

- 毫秒级突触时延和高频振荡；
- 尖峰同步或膜时间常数；
- 无标签条件下的单细胞 E/I 身份；
- 未经时间反转、surrogate 或干预验证的有效连接方向。

---

旧项目说明来源：[SRHarness neural README](https://github.com/yuzhTHU/SRHarness/tree/pre-refactor/src/experimental/neural)。
