你现在扮演一名人工智能领域资深的「论文复现技术顾问」，目标是帮助我**完整、可落地地复现一篇算法论文**，包括：

- 明确问题与目标指标
- 拆解论文中的技术模块
- 结合我给定的本地环境、数据路径、参考代码仓库
- 产出：分阶段实现方案、模块拆分、函数级 TODO、验证与可视化策略

请严格遵守以下要求：

- 回答使用中文
- 给出结构化输出（使用一级/二级标题）
- 不直接写大段源码，而是给出**模块划分 + 函数级设计 + 伪代码/接口示例**
- 强调「如何验证正确性」（可视化/指标/对齐论文）

## 一、论文与环境信息

### 1.1 论文基本信息（已给定）

- 标题：`Flux4D: Flow-based Unsupervised 4D Reconstruction`
- 作者 / 会议 / 年份：`NeurIPS 2025`
- 原文 PDF 路径：`docs/论文中202512_Flux4D.pdf`
- 中文翻译文档：`Flux4D-基于光流的无监督4D重建-翻译.md`
- 官方代码：`目前公开信息显示无官方代码仓库`
- 相关技术栈关键词：`3D Gaussian Splatting, 稀疏 3D U-Net, LiDAR + 多相机, 光流/场景流, 动态重加权, 无监督 4D`

### 1.2 我的本地环境与约束（已给定）

- 目标：在 PandaSet 上复现 Flux4D 的主要指标（PSNR/SSIM/Depth RMSE/Flow EPE 级别接近论文），并完成可视化效果
- 主要数据集：
  - 名称：`PandaSet`
  - 本地路径：`/home/yr/yr/data/automonous/pandaset`
  - 后续扩展：`Waymo`（暂未下载）
- 主要参考代码仓库：
  - `GaussianSTORM`: `/home/yr/yr/code/cv/AutoLabel/SSL/GaussianSTORM_all/GaussianSTORM`
    - 环境：`/home/yr/anaconda3/envs/storm`
    - 已有：gsplat 渲染接口、数据管线、时空 Transformer、sky token 等
- 运行硬件：
  - 假设：`1×5090 24G` GPU

请你先**简要复述你对任务与约束的理解**，然后再展开后续分析。

---

## 二、论文核心技术拆解（请你完成）

重点结合 Flux4D 的特点回答：

1. 任务定义

   - 输入：多帧 LiDAR 点云 + 多相机 RGB 图像 + 外参/时间戳
   - 输出：一组带速度的 3D 高斯（4D 场景表示），可在任意时间与视角下渲染 RGB/Depth/Flow
   - 核心特性：无监督、flow-based 4D reconstruction、无需语义标注
2. 整体框架模块

   - Lift：LiDAR 点 → 初始 3D 高斯（位置、尺度、方向、颜色、opacity、时间 t_i、初始速度 0）
   - Predict：基于 3D 稀疏 U-Net / Transformer 的网络，输入 G_init，输出每个高斯的更新参数 + 3D 速度 v_i
   - Render：时间推进 p_i(t') = p_i + v_i·(t'-t_i)，使用 3DGS 渲染 RGB/Depth/Flow
   - Loss：光度损失（L1 + SSIM）+ 深度损失 + 速度 L2 正则 + 基于渲染flow的动态重加权
3. 损失项与数学形式

   - 请你列出：L_recon, L_SSIM, L_depth, L_vel, L_dyn_weighted 等的公式与直观含义
4. 和 GaussianSTORM 的对应

   - 指出：
     - 3DGS 渲染部分如何复用 GaussianSTORM 已有模块
     - 时空编码/Transformer 如何在 STORM 基础上改造为 Flux4D 风格（从「未来预测」转为「速度+4D重建」）

---

## 三、结合我环境的实现方案（分阶段路线图）

请你给出一个具体、可执行的阶段划分，至少包含：

1. 阶段 0：环境与基线验证

   - 在 `storm` 环境下跑通 GaussianSTORM 的 demo
   - 确认 gsplat/3DGS 渲染后端可用
2. 阶段 1：PandaSet 数据标签与 tiny/full 索引

   - 生成：
     - `data/metadata/pandaset_full_clips.pkl`
     - `data/metadata/pandaset_tiny_clips.pkl`（1~2 个场景，用于调试）
   - 要求：
     - 每条 clip 明确记录：帧时间戳、相机/雷达外参、图像/点云路径
3. 阶段 2：Lift 模块实现与验证

   - 在 `lift/` 下实现：
     - LiDAR 体素下采样 + kNN 距离估计
     - LiDAR → 高斯（位置/尺度/颜色/opacity/t_i/vel=0）
   - 可视化门禁：
     - 初始高斯渲染 vs 原图
     - 点数 & 高斯数统计
4. 阶段 3：基于 GaussianSTORM 的编码器改造

   - 复用它的：
     - 数据管线
     - Transformer/Perceiver 类结构（如果合适）
   - 将输入从「图像 token」改为「体素化的高斯特征」，输出为「点级特征」
5. 阶段 4：速度分支 + 时间推进

   - 对每个高斯预测 3D 速度 v_i
   - 使用线性运动模型 p_i(t') = p_i + v_i (t'-t_i)
   - 在渲染前调用时间推进
6. 阶段 5：光度损失 + 深度损失 + 速度正则 + 动态重加权

   - 组织训练循环：
     - 渲染 t0/t1/t' 的 RGB/Depth/Flow
     - 计算带动态权重的 photometric loss
     - 计算 L_vel 正则
7. 阶段 6：全量训练与评测

   - 在 PandaSet full 上训练
   - 输出指标：PSNR/SSIM/Depth RMSE/Flow EPE，与论文进行量级对比
   - 可视化典型场景的 4D 重建视频

---

## 四、模块–文件–函数级 TODO 列表（请你输出）

请仿照下面结构，把 Flux4D 需要的模块拆成「模块 → 文件 → 函数」，并简要写出输入/输出及注意事项。

示例（请扩展）：

- 模块：数据与索引（PandaSet）

  - 文件：`src/flux4d/datasets/pandaset_clips.py`
    - 函数：`build_pandaset_clip_index(data_root, out_pkl_full, out_pkl_tiny)`
    - 函数：`load_clip(meta_entry) -> ClipData`
- 模块：LiDAR → 高斯（Lift）

  - 文件：`src/flux4d/lift/lift_lidar.py`
    - 函数：`voxel_downsample_points(...)`
    - 函数：`compute_knn_mean_distance(...)`
    - 函数：`project_point_to_cameras(...)`
    - 函数：`create_gaussian_from_point(...)`
    - 函数：`build_initial_gaussians_for_clip(...)`
- 模块：主网络（Flux4DModel）

  - 文件：`src/flux4d/storm/flux4d_model.py`
    - 函数：`forward(G_init) -> (G_refined, velocities, aux)`
- 模块：渲染与光流

  - 文件：`src/flux4d/render/flux4d_renderer.py`
    - 函数：`apply_linear_motion(G_refined, velocities, t_target)`
    - 函数：`render_rgb_depth(...)`
    - 函数：`render_flow_map(...)`
- 模块：损失与动态重加权

  - 文件：`src/flux4d/losses/flux4d_losses.py`
    - 函数：`photometric_l1_loss(...)`
    - 函数：`ssim_loss(...)`
    - 函数：`velocity_regularization(...)`
    - 函数：`build_dynamic_weight_map_from_flow(...)`
    - 函数：`total_loss(...)`

请你完整列出这些 TODO。

---

## 五、验证与可视化策略（请你单独列出）

1. 对于每个阶段，画什么图、存什么中间结果来验证？
2. 对于 Flux4D 特有的 4D 动态场景，如何生成「时间序列视频」用于主观评估？
3. 如何用小的 tiny 数据集快速验证模型是否在朝正确方向学习（例如过拟合一个 clip）？

---

## 六、风险点与优先级（请你总结）

列出至少 5–10 个关键风险点（比如数据对齐、显存、训练不稳定等），并给出对应的对策和建议优先级执行顺序。

---

## 七、最终输出形式

请将你的回答组织成以下几个部分：

1. 任务与约束复述（简短）
2. 论文技术拆解
3. 分阶段实现路线图
4. 模块–文件–函数级 TODO 列表
5. 验证与可视化策略
6. 风险与优先级

文风偏工程说明，可直接作为项目文档和 issue 使用。

我上面的信息，除了“一、论文与环境信息"这部分信息，其它部分你都要仔细阅读论文，分析是否我错误或者需要补全的部分。保证最终输出的形式部分的结果与论文一致。

现在，开始分析并给出你的回答，将回答放在docs/开发记录/复现计划.md中
