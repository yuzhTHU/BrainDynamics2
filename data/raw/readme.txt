wholebrain_output_after_rejection.mat 
- 包含了神经元的发放信息，变量whole_trace_ori （NxT)中是N个神经元T个时间采样步的发放率信息

valid_area.mat
- 包含了N个神经元在皮层上的3维坐标和脑区归属信息（不随时间变化）

stimuli.mat
- 包含了状态切换的信息
- 变量stimuli.stimuili_label_ind是每个状态的id（1 for REM, 2 for Awake, 3 for NREM)，标签由头皮脑电(EEG）给出，可以认为是真值
- 变量stimuli.start_edge是每个状态开始的帧数
- 变量stimuli.end_edge是每个状态结束的帧数

视频是同步记录的小鼠面部信息（暂未时间对齐，可对齐）
