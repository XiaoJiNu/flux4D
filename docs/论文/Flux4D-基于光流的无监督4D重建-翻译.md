# Flux4D：基于光流的无监督4D重建

**作者：**
Jingkang Wang$^{1,2*}$, Henry Che$^{1,3*\dagger}$, Yun Chen$^{1,2*}$, Ze Yang$^{1,2}$
Lily Goli$^{1,2\dagger}$, Sivabalan Manivasagam$^{1,2}$, Raquel Urtasun$^{1,2}$

**机构：**
$^1$Waabi, $^2$多伦多大学 (University of Toronto), $^3$伊利诺伊大学香槟分校 (UIUC)

**项目主页：** https://waabi.ai/flux4d

---

## 摘要 (Abstract)

从视觉观测中重建大规模动态场景是计算机视觉中的一项基础性挑战。虽然最近的可微渲染方法（如NeRF和3DGS）取得了令人印象深刻的照片级逼真重建效果，但它们受限于可扩展性，并且需要标注来将移动的主体（Actor）与静态场景解耦，例如在自动驾驶场景中。现有的自监督方法试图通过利用运动线索和几何先验来消除显式标注，但它们仍然受限于逐场景（per-scene）优化以及对超参数调整的敏感性。在本文中，我们介绍了 **Flux4D**，这是一个简单且可扩展的框架，用于大规模动态驾驶场景的4D重建。Flux4D直接预测3D高斯（3D Gaussians）及其运动动力学，以完全无监督的方式重建传感器观测数据。通过仅采用光度损失（photometric losses）并强制执行“尽可能静止（as static as possible）”的正则化，Flux4D学会了直接从原始数据中分解动态元素，而无需预训练的监督模型或基础先验，这仅仅通过在许多场景中进行训练即可实现。我们的方法能够在几秒钟内高效地重建动态场景，有效地扩展到大型数据集，并能很好地泛化到未见过的环境中，包括稀有和未知的物体。在户外驾驶数据集上的实验表明，Flux4D在可扩展性、泛化能力和重建质量方面显著优于现有方法。

---

## 1. 引言 (Introduction)

从野外捕获的视觉观测中重建4D物理世界是计算机视觉的一个关键目标，在虚拟现实和机器人技术（包括自动驾驶）中都有应用。高质量的重建为可扩展的仿真环境提供了基础，从而实现更安全、更高效的自动驾驶开发。与艺术家创建的环境不同，利用传感器配备车辆收集的数据自动构建的环境更加逼真、成本效益更高，并且捕捉到了现实世界的多样性 [50, 62, 31]。

可微渲染方法的进步，如神经辐射场 (NeRF) [32] 和 3D高斯泼溅 (3DGS) [18]，已经实现了动态场景的高质量重建 [62, 59, 72, 45, 19, 6, 46]。这些方法使用人工标注（如3D轨迹片段或动态掩码）将场景分解为静态背景和一组动态主体，然后对组合的表示进行渲染，通过优化来重建输入观测。虽然它们实现了令人印象深刻的视觉保真度，但它们依赖人工标注来分解静态和动态元素，这增加了成本和时间，阻碍了这些方法扩展到大量未标记数据集。一些方法利用预训练的感知模型自动生成标注，但当模型预测有噪声或不正确时，这会导致伪影，且在重建过程中很难恢复。此外，这些方法通常需要数小时才能在消费级GPU上重建每个场景。昂贵的标注成本和缓慢的逐场景优化这两个主要问题限制了这些方法的可扩展性。

最近的工作探索了自监督方法，以消除对人工标注的依赖，并直接从数据中学习静态和动态主体的分解。由于随时间变化的物体运动的模糊性，加上空间几何和外观的变化，这是一项具有挑战性的任务。一种策略试图通过引入额外的正则化项（如几何约束 [37] 或循环一致性 [61]）或执行多阶段训练 [17] 来改进分解。另一种策略是利用基础模型获得额外的语义特征或先验 [37, 8, 61]。然而，由此产生的复杂模型可能对超参数敏感，训练缓慢，并且无法泛化到新场景。此外，它们的分解结果往往较差，并且难以渲染新视角，限制了其实用性。

作为昂贵的逐场景优化的替代方案，可泛化（generalizable）的方法 [3, 51, 2, 5, 14, 53, 69] 使用前馈神经网络直接从观测中预测场景表示，从而在几秒钟内实现高效重建。然而，这些方法是为小规模环境设计的，只能处理少量低分辨率图像（通常为1-4个视图，分辨率低于512px），并且主要关注静态场景 [2, 5] 或仅关注动态对象 [40]。在处理具有许多动态元素的大型场景时，它们依赖于昂贵的标注 [7, 41]，限制了其可扩展性。最近，DrivingRecon [30] 和 STORM [60] 提出了针对驾驶场景的前馈、自监督方法。虽然很有前景，<u><!--但这些方法专注于稀疏重建设置，并且在达到计算极限之前只能处理少量（≤ 12）低分辨率（≤ 360px）的输入视图--></u>，并且<u>仍然依赖于预训练的视觉模型</u>进行语义引导，从而限制了它们的保真度、可扩展性以及在下游仿真中的适用性。

在本文中，我们提出了 **Flux4D**，这是一种**无监督**且**可泛化**的重建方法，能够大规模地实现准确且高效的4D驾驶场景重建。Flux4D 无需任何标注，即可在几秒钟内从多传感器观测中直接在3D空间中预测3D高斯及其运动参数，从而实现高效的场景重建。我们的重建范式如图1所示。Flux4D 使用极其简简的设计，仅采用光度损失和简单的静态偏好先验，而无需先前工作所利用的复杂正则化方案或外部监督来学习运动。我们发现，Flux4D 准确恢复几何、外观和运动流的关键要素来自于跨各种场景的学习。此外，Flux4D 利用了自动驾驶领域通用的 LiDAR 数据，能够处理大量（≥ 60）高分辨率（1080px）输入多视图图像，实现高保真重建和可扩展仿真。我们的3D设计产生了跨视图紧凑且几何一致的表示，提高了效率，实现了显式的多视图光流推理，并减少了外观-运动的模糊性。

在户外驾驶数据集 PandaSet [57] 和 WOD [42] 上的实验表明，Flux4D 实现了比以前最先进的无标注重建方法更好的场景分解和新视角合成，并且与使用人工标注的逐场景优化方法具有竞争力。我们还表明，<u>Flux4D 可以被训练来预测未来帧中的传感器观测结果，类似于“下一个token预测”，但应用于动态3D场景。</u>最后，我们展示了使用 Flux4D 的重建结果进行可控的相机仿真（通过场景编辑）和高分辨率（≥ 1080px）的新视角渲染。Flux4D 突显了无监督学习在4D场景重建中的力量，实现了向海量未标记数据集的高效扩展。

---

## 2. 相关工作 (Related Work)

**基于优化的4D重建：** 受可微渲染 [32, 18] 的启发，最近的方法使用变形场 [38, 36, 66, 55] 来模拟动态场景，但由于过参数化和较差的静态-动态分解，仍然难以应对现实世界的复杂性。虽然有些方法通过使用人工标注（3D轨迹、语义模型）显式分离静态 [49, 63] 和动态元素 [35, 62, 29, 47, 39, 59, 10, 13, 64] 来解决这个问题，但它们仍然受限于标注质量和可用性。使用运动线索和物理信息先验的自监督替代方案 [56, 61, 8, 17, 37] 减少了对标注的依赖，但通常需要复杂的正则化方案和昂贵的逐场景优化。相比之下，我们的方法在没有显式监督或逐场景优化的情况下重建动态4D场景，通过简单的光度损失和最小的正则化实现了可扩展的重建。

**可泛化的重建：** 可泛化方法直接从观测中推断场景表示，无需逐场景优化 [3, 51, 2, 5, 14, 53, 69]，利用大型训练数据集来提高新环境中的重建质量。然而，现有方法主要针对静态场景，由于计算限制和依赖稀疏、低分辨率输入，难以应对动态环境。最近的进展尝试使用高效架构 [73] 或迭代优化 [7] 来克服这些限制，但仍依赖于3D标注。相比之下，Flux4D 通过直接从原始观测中预测带有运动的3D高斯，在没有外部监督的情况下泛化到未见过的动态场景。

**无监督世界模型：** 我们的工作与无监督世界模型的最新进展有关，这些模型在没有显式监督的情况下学习环境的预测性表示。这些方法通常将视觉数据标记化为离散或连续的表示 [15, 12, 52, 71, 33]，由自回归或基于扩散的模型处理以预测未来状态。虽然展示了令人印象深刻的视觉质量，但<u>此类方法通常缺乏可解释的3D结构</u>，限制了对生成内容的精确控制。现有的解决方案通常产生较低分辨率的输出，时间一致性降低，通常局限于单一模态（例如，相机 [15, 12, 26] 或 LiDAR [21, 70, 65, 1]），并且需要大量的计算资源。虽然我们的主要重点是重建，但 Flux4D 同时建模运动动力学和预测未来帧的能力与世界模型在概念上有相似之处。与这些方法不同，Flux4D 使用显式的3D表示，提供3D可解释性、可控性和时空一致性。

疑问：这里的运动动力学是怎么建模的？

疑问：如果不想用相机的内外参，直接用模型学习相机内外参数可以吗，怎么做？

**无监督可泛化重建：** 最近，DrivingRecon [30] 和 STORM [60] 探索了驾驶场景的无监督可泛化4D重建，使用前馈网络预测3D高斯的速度。尽管性能令人印象深刻，但它们只能处理稀疏（3-4帧）、低分辨率（≤ 256 × 512）的帧，计算要求很高，并且依赖预训练的视觉模型（DeepLabv3+ [4], SAM [23], ViT-Adapter [9]）进行额外监督，限制了它们的可扩展性和适用性。Flux4D 通过更简单、更可扩展的方法实现了更好的性能，并且通过我们新颖地结合LiDAR来初始化场景，可以在计算高效的同时处理具有更密集视图（> 60）的全高清（Full HD）图像。更多讨论请参阅补充材料。

疑问：如果没有lidar数据，flux4d应该怎么训练呢？

---

## 3. 基于Flux4D的可扩展4D重建

给定机器人传感器平台捕获的一系列相机和LiDAR数据，我们的目标是重建潜在的4D场景表示，该表示能够解耦静态和动态实体，并支持在新视角下的高质量渲染。<u>这种表示可以实现未来预测和反事实模拟</u>。为了实现可扩展的4D场景重建，我们的方法应该是**无监督的**（意味着不使用标注），并且是**快速的**（在几秒钟内运行）。为此，我们提出了 Flux4D，这是一种无监督且可泛化的方法，学习通过三个简单的步骤重建4D场景（图2）。我们首先将每个时间步的传感器观测提升为一组初始3D高斯。然后，我们将初始表示输入到一个网络中，以预测每个3D高斯的3D光流和精炼属性。最后，我们仅通过重建和静态偏好损失来监督网络。

疑问：什么是这种表示可以实现未来预测和反事实模拟？counterfactual simulation是什么？

疑问：不同视角的不同像素对应空间中的同一个高斯核，这种情况是怎么处理的？

### 3.1 场景表示

我们的方法采用一组带有位姿的相机图像 $\mathcal{I} = \{I_k\}_{1\leq k \leq K}$ 和 LiDAR 点云 $\mathcal{P} = \{P_k\}_{1\leq k \leq K}$，这些数据是由移动平台随时间捕获的，并输出具有几何、外观和3D光流的场景表示。我们将场景表示为一组3D高斯 $\mathcal{G} = \{g_i\}_{1\leq i \leq M}$。每个高斯点 $g_i$ 由其中心位置 $\mathbf{p}_i (\mathbb{R}^3)$、缩放 $(\mathbb{R}^3)$、方向 $(\mathbb{R}^4)$、颜色 $(\mathbb{R}^3)$ 和不透明度 $(\mathbb{R}^1)$ 参数化 [18]。此外，我们为每个高斯增强了一个可学习的瞬时速度 $\mathbf{v}_i \in \mathbb{R}^3$ 和一个固定的捕获时间 $t_i$。我们将所有高斯的速度集和时间戳集表示为 $\mathcal{V} = \{\mathbf{v}_i\}_{1\leq i \leq M}$ 和 $\mathcal{T} = \{t_i\}_{1\leq i \leq M}$。

疑问：捕获时间 $t_i$是什么，怎么得到的？

**初始化：** 我们从序列中每个源帧的 LiDAR 点 $P_k$ 初始化高斯位置，根据到附近点的平均距离设置缩放比例，并通过将这些点投影到相应的相机图像 $I_k$ 上来分配颜色。每个高斯的时间戳 $t_i$ 被分配为其源 LiDAR 帧的捕获时间，速度初始化为零。我们聚合源帧高斯以创建 $\mathcal{G}_{\text{init}}$。

疑问：这里初始化的高斯核数量和点云数量一样多是吗？那每次学习的高斯核的数量是不同的对吗？模型怎么处理这种数量不同的数据呢？

疑问：如果点云稀疏，那么学习到的高斯核是不是就比较稀疏，那这个场景重建的效果是不是就受限于点云的密度？是需要多帧叠加得到稠密的点云吗？

疑问：根据到附近点的平均距离设置缩放比例，这个具体怎么做呢？

### 3.2 预测光流与渲染

受最近4D重建进展 [56, 61, 37, 68, 30, 60] 的启发，我们建议**学习一个时间相关的速度场来模拟驾驶场景的动力学**。给定初始的速度增强高斯 $\mathcal{G}_{\text{init}}$，我们利用一个神经重建函数 $f_\theta$ 输出精炼的高斯参数 $\mathcal{G}$ 和预测的速度 $\mathcal{V}$：
$$ \mathcal{G}, \mathcal{V} = f_\theta(\mathcal{G}_{\text{init}}, \mathcal{T}). \quad (1) $$

有了预测的速度 $\mathcal{V}$，每个高斯可以使用线性运动模型从其初始时间步 $t_i$ 传播到任何目标时间步 $t'$：
$$ \mathbf{p}_i^{t'} = \mathbf{p}_i^{t_i} + \mathbf{v}_i \cdot (t' - t_i), \quad (2) $$
其中 $\mathbf{p}_i^{t'}$ 是时间 $t'$ 时的高斯位置，$\mathbf{v}_i$ 和 $t_i$ 是其速度和捕获时间。这种公式使得在恒定速度假设下能够实现连续、时间一致的重建。我们发现，在重建具有短时间范围（~1秒）的户外驾驶场景时，这种简单的运动模型已经可以取得合理的性能，这一观察与现有工作 [37, 30, 24, 60] 一致。此外，我们在第3.4节和表7中讨论了更高阶的多项式运动模型。

疑问：这里预测的速度，表示的是该时刻的速度是吗？如果t1,t2时刻速度分别为v1，v2。那用平均速度是否更合理？

### 3.3 动力学的无监督学习

现在我们描述该方法如何学习解耦场景动力学。网络 $f_\theta$ 以完全自监督的方式训练，无需显式的3D标注。给定预测的高斯 $\mathcal{G}$，我们利用公式(2)将高斯移动到目标时间 $t'$，使用可微光栅化 [18] 渲染场景以生成彩色和深度图像，并将它们与真实的传感器观测 $\mathcal{I}$ 和 $\mathcal{P}$ 进行比较。为了防止不必要的运动并鼓励稳定性，我们引入了“尽可能静止（as static as possible）”的正则化。总损失 $\mathcal{L}$ 定义为：
$$ \mathcal{L} = \mathcal{L}_{\text{recon}} + \lambda_{\text{vel}}\mathcal{L}_{\text{vel}}, \quad (3) $$
其中 $\mathcal{L}_{\text{recon}}$ 代表重建损失，包括关于**图像的 $L_1$ 和结构相似性（SSIM）损失**，以及图像平面上相对于**投影LiDAR的 $L_1$ 深度损失**；$\mathcal{L}_{\text{vel}}$ 作为速度正则化项，用于最小化运动幅度：
$$ \mathcal{L}_{\text{recon}} = \lambda_{\text{rgb}}\mathcal{L}_{\text{rgb}} + \lambda_{\text{SSIM}}\mathcal{L}_{\text{SSIM}} + \lambda_{\text{depth}}\mathcal{L}_{\text{depth}}, \quad (4) $$
$$ \mathcal{L}_{\text{vel}} = \frac{1}{M} \sum_i \|\mathbf{v}_i\|_2. \quad (5) $$

我们在各种场景中训练 $f_\theta$。值得注意的是，我们发现**跨许多场景训练**使网络能够自动分解城市场景中的静态和动态成分，而无需以前逐场景优化技术 [56, 61, 8, 17, 37] 中使用的复杂正则化。这突出了数据驱动先验作为一种强大的隐式正则化形式的有效性，以及这种简单框架的可扩展性。

疑问：光流是什么？光度是什么？

### 3.4 提高真实感和光流 (Improving Realism and Flow)

上述组件构成了我们方法的核心，称为 *Flux4D-base*。Flux4D-base 已经可以高质量地解耦运动并渲染新视角。我们通过两项增强进一步改进 Flux4D-base，恢复更细粒度的外观和精炼的光流，从而得到我们的最终模型 *Flux4D*。

**迭代优化 (Iterative refinement):** Flux4D-base 恢复了整体场景外观，但往往缺乏细粒度的细节。我们假设这种局限性源于单步前馈网络的受限容量，以及由于遮挡导致的初始化不完美。为了缓解这一问题，我们引入了一种受 G3R [7] 启发的迭代优化机制，利用**3D梯度**作为反馈来提高重建质量。具体来说，在每次前向传播并在监督视图生成渲染的颜色和深度之后，我们根据损失函数公式(3)计算高斯的3D梯度，并将生成的高斯和梯度作为输入提供给网络 $f_\phi$ 以进一步优化它们。这个过程逐步修正颜色不一致并锐化细节，最少只需两次迭代。通过结合迭代反馈，我们的方法实现了更高保真度的重建，特别是在外观变化复杂的区域，同时保留了 Flux4D-base 的效率和可扩展性。

疑问：3D梯度是什么？将生成的高斯和梯度作为输入提供给网络 $f_\phi$ 以进一步优化它们。这个应该怎么做？

**运动增强 (Motion enhancement):** Flux4D-base 准确地恢复了整体场景光流（表7）。我们进一步引入<u>多项式运动参数化</u>，以更好地模拟加速、刹车或转弯等主体行为。更多细节和比较请参见补充材料。探索更先进的速度模型 [24] 或隐式流表示是未来工作的一个令人兴奋的方向。为了进一步提高动态主体的光流和外观质量，我们修改了损失函数以关注动态区域。具体来说，我们在图像平面渲染光流，并对光度损失应用逐像素重加权。这在训练期间赋予移动较快的区域更高的重要性，这些区域通常占据较少的像素，对整体损失的贡献较小。

---

## 4. 实验 (Experiments)

我们将 Flux4D 与当前最先进的（SoTA）自监督场景重建方法进行评估，包括逐场景优化方法和可泛化方法。作为参考，我们还报告了需要标注来模拟动力学的监督方法的性能。我们在多个户外动态数据集上进行实验，评估新视角的外观和深度，以及恢复的光流。我们还对 Flux4D 的设计进行了消融实验，并展示了 Flux4D 随更多数据扩展的能力。最后，我们展示了我们的预测场景表示在逼真相机仿真中的可控性。

### 4.1 实验细节

**实验设置：** 我们在 PandaSet [57] 和 Waymo Open Dataset (WOD) [42] 的户外驾驶场景上进行实验。从 **PandaSet** 的 103 个动态场景（1080p 相机，64线 LiDAR，10Hz）中，我们选择 10 个不同的场景进行验证，其余用于训练。我们<u>使用前置相机和 360° LiDAR，均以 10 Hz 采集。</u>为了与现有的只能处理少量输入帧的前馈可泛化重建方法进行比较，我们报告验证序列中 1.5秒短窗口的场景重建结果。每种方法以帧 0, 2, 4, 6, 8, 10 作为输入，并在帧 1, 3, 5, 7, 9（插值）和 11-15（未来预测）上进行评估。我们每20帧采样一个新的片段，每个日志产生四个不重叠的评估片段。我们还在插值设置下（每隔一帧保留作为测试），在验证序列的全时长（8秒）上评估逐场景优化方法。对于 **WOD 评估**，我们遵循 DrivingRecon [30] 中的 NVS 设置，<u>使用具有三个前置相机的 Waymo-NOTR 子集</u>，采用 $\{t-2, t-1, t+1\}$ 帧作为输入，并生成时间 $t$ 的插值帧，其中 $t$ 是每个序列中的每第十帧。最后，我们在 PandaSet 和 WOD（官方验证集，包含202个日志）上评估场景流估计性能。由于现有的场景流估计方法无法直接预测新时间步的光流，我们在输入帧上评估场景流。<u>我们限制在相机视场（FoV）内的 LiDAR 点上进行评估，遵循 [60]</u>。

疑问：每种方法以帧 0, 2, 4, 6, 8, 10 作为输入，并在帧 1, 3, 5, 7, 9（插值）和 11-15（未来预测）上进行评估。这里为什么是插值，预测又是怎么得到的？

**基线方法：** 我们与 SoTA 无监督场景重建方法进行比较：
(1) *自监督逐场景优化：* EmerNeRF [61] 和 DeSiRe-GS [37]，利用几何先验、循环一致性和预训练的视觉模型（FiT3D [67] 和 DINOv2 [34]）重建动态场景；
(2) *可泛化方法：* L4GM* [40]，一种使用深度监督适配于驾驶场景的4D重建模型；DepthSplat*，[58] 的扩展，利用估计深度反投影 LiDAR 点进行 3D 高斯预测；DrivingRecon [30]，建立了一个利用预训练视觉模型（SAM [23] 和 DeepLab-v3 [4]）的学习先验的 4D 前馈模型；以及 STORM [60]，以前馈方式预测逐像素高斯及其运动。作为参考，我们还包括使用地面真值 3D 轨迹片段的 SoTA 方法：StreetGS [59] 和 NeuRAD [45]（组合式 3DGS/NeRF），以及 G3R [7]（组合式 3DGS 的迭代优化）。除了重建方法外，我们还与代表性的场景流估计方法 NSFP [27] 和 FastNSF [28] 进行比较作为参考。

疑问：以及 STORM [60]，以前馈方式预测逐像素高斯及其运动。STROM的逐像素高斯是怎么实现的，它和flux4D的高斯实现方式有什么区别？

疑问：在pandaset中只用了前视相机，点云是360的，那此时只有前视相机才能都督，那最终重建的效果在前视相机以外的区域效果是不是很差？

**指标：** 我们报告标准指标来衡量照片级真实感、几何和运动精度，使用 PSNR、SSIM 以及深度 RMSE ($D_{\text{RMSE}}$) 和速度 RMSE ($V_{\text{RMSE}}$)。结果在全图和动态移动区域上报告，以进行综合评估。对于场景流质量，我们报告 EPE3D、$Acc_5$ 和 $Acc_{10}$（误差 $\le 5/10$ cm 的点比例）、弧度角误差 ($\theta_\epsilon$)、三路 EPE [11]：背景-静态 (BS)、前景-静态 (FS) 和前景-动态 (FD)、分桶归一化 EPE [20] 以及推理速度。在 WOD 上，由于语义标签较粗糙，我们遵循 EulerFlow [48] 并在*背景（包括标志）*、*车辆*、*行人*和*骑行者*上报告分桶归一化 EPE。

**Flux4D 实现细节：** 我们采用带有稀疏卷积的 **3D U-Net [43] 作为 **$f_\theta$。为了处理<u>无界场景</u>，我们将随机点放置在远处的球面上以模拟天空和远方区域。我们还遵循 [59] 在 3D 球体内添加随机点以增加模型鲁棒性。我们的模型在所有实验中处理全分辨率图像（$\ge 1920 \times 1080$），并且可以有效地扩展到更高分辨率而没有显著的开销。除非另有说明，所有模型均在 4× NVIDIA L40S (48G) GPU 上训练 30,000 次迭代，大约耗时 2 天。重建损失权重 $\lambda_{\text{rgb}}, \lambda_{\text{SSIM}}, \lambda_{\text{depth}}$ 分别设置为 0.8, 0.2 和 0.01。速度正则化权重 $\lambda_{\text{vel}}$ 设置为 5e-3。

疑问：无界场景是什么？我们将随机点放置在远处的球面上以模拟天空和远方区域中的随机点和远程的球面是什么？

疑问：3D 球体是什么？

### 4.2 可扩展的 4D 重建结果

**PandaSet 上的新视角合成：** 表1 和图3 在插值设置下的 1s PandaSet 片段上比较了 Flux4D 与 SoTA 无监督方法，并包括监督方法作为参考。重建速度是在单个 RTX A5000 GPU (24GB) 上测量的。Flux4D 实现了卓越的照片级真实感和几何精度，并具有快速的重建速度。我们进一步评估了我们的方法在更长视野（8秒日志）重建上的表现（表2 和 图4），使用 1s 片段的迭代处理。我们的方法在 1s 和 8s 重建任务上都大幅优于无监督的逐场景优化方法，且无需预训练模型或复杂的正则化。我们的定量结果也表明，Flux4D 即使与监督方法相比也具有竞争力。定性来看，如图3和4所示，Flux4D 在静态和动态区域都实现了高保真的相机渲染，而现有的无监督方法通常由于学到的动力学不准确，在动态主体上会出现明显的伪影。

**WOD 上的新视角合成：** 我们进一步在表3中将 Flux4D 与 WOD 上的 SoTA 可泛化方法进行比较，设置遵循 [30]。基线结果来自 DrivingRecon [30] 论文，我们与作者确认了设置和结果以确保准确比较。Flux4D 在 PSNR 上超过 DrivingRecon +5.99 dB，在 SSIM 上超过 +0.21，证明了其在无监督动态场景重建方面的有效性。定性比较请参阅补充材料。

疑问：PSNR, SSIM, LPIPS，这些指标分别是什么？

**光流估计：** 我们将 Flux4D 的估计运动流与现有的无监督逐场景优化方法 EmerNeRF [61] 和 DeSiRe-GS [37] 进行比较。如表1、2和图5所示，Flux4D 显著优于以前的方法，在没有任何监督的情况下学习到了准确的运动方向和幅度。相比之下，现有方法难以学习一致的运动流并完全分解动态场景，导致运动预测不准确和不连贯，限制了它们在下游任务中的适用性。

疑问：运动流是什么？

**场景流评估：** 虽然 Flux4D 主要专注于重建，并非专门为场景流估计设计，但我们进一步在 PandaSet 上评估了其性能，并在表5和6中与代表性的场景流估计方法使用标准场景流指标进行了比较。关于 WOD 的比较请参阅补充材料。尽管不是为场景流估计设计的，但 Flux4D 仅使用基于重建的监督（RGB + 深度）就在大多数场景流指标上实现了卓越的性能。值得注意的是，它在较小或不太常见的对象类别（如轮式弱势道路使用者(VRU)、其他车辆和行人）上优于其他方法，如分桶评估所示。这些结果凸显了一条在单一框架内统一最先进场景流估计 [20, 20, 22, 25] 和重建的有前途的途径。

**未来预测：** 我们评估 Flux4D 预测观测帧之外的未来帧的能力。这项具有挑战性的任务需要精确的运动估计、时间一致性、遮挡推理和全面的 4D 场景理解。如表4所示，Flux4D 在光度精度和几何一致性方面均优于现有的无监督方法。此外，Flux4D 甚至优于依赖不完美的显式标注进行外推的监督方法，证明了我们预测的场景表示的鲁棒性和无监督场景流预测的有效性。这突出了 Flux4D 模拟场景动力学的能力，这对自动驾驶系统中的世界建模、仿真和场景理解至关重要。我们在表4中报告了仅动态（dynamic-only）的指标，全图指标参见补充材料。

疑问：几何一致性是什么？这里的几何是什么？

疑问：补充材料在哪？

**消融实验：** 表7评估了 Flux4D 的关键设计组件。迭代优化显著提高了图像质量和几何精度指标。多项式运动建模提高了运动预测性能。表8证明了我们的静态偏好先验对于学习准确的光流至关重要，而速度重加权提高了动态元素的性能。定性比较请参阅补充材料。

**无 LiDAR 的 Flux4D：** 我们表明 Flux4D 也可以在<u>推理时以无 LiDAR 模式运行</u>，类似于 DrivingRecon [30] 和 STORM [60]，方法是使用现成的单目深度估计模型 [16]。如表9所示，光流估计性能保持相当，在某些情况下，由于单目深度提供了更广泛的覆盖范围（例如建筑物），背景区域的视觉真实感有所提高，特别是在 LiDAR 稀疏限制重建质量的区域。结合 LiDAR 和由单目深度提升的点可以产生最佳的整体真实感。

疑问：方法是使用现成的单目深度估计模型。这个具体怎么做的？

**扩展分析：** Flux4D 的有效性源于多场景训练，利用多样化的驾驶数据作为隐式正则化。与需要复杂正则化或预训练模型的逐场景方法不同，增加训练数据量自然会改善场景分解和运动估计。在 PandaSet 和 WOD 上的分析表明，随着训练规模的扩大，光度精度和运动估计持续改进。这证实了无监督 4D 重建显著受益于多样化的现实世界场景，表明 Flux4D 可以随着更多数据继续改进，使其有望用于可扩展的场景重建。

**相机仿真：** 我们展示了应用 Flux4D 在大规模驾驶场景中进行高保真相机仿真。Flux4D 在 PandaSet（图6）、Argoverse 2 [54] 和 WOD（图7）上的多样化、大规模动态场景中产生了高质量的运动流。这允许跨不同环境进行准确的场景分解，这对实例提取和动态元素的直接操作至关重要（图9）。与现有的自监督逐场景方法相比，Flux4D 更适合交互式和可控应用，因为它重建了一个可编辑的表示，支持实例掩码提取、场景编辑和各种下游任务的对象操作。在图9中，我们展示了 Flux4D 渲染修改后的场景表示的逼真图像的能力。值得注意的是，我们的方法无需标签即可实现这一点。

疑问：这里既然支持实例掩码提取，是不是就可以得到每个目标的3D box和轨迹？

---

## 5. 局限性 (Limitations)

尽管 Flux4D 在没有任何标注或预训练模型的情况下实现了 SoTA 的 4D 重建，但仍存在三个关键限制：(1) 对具有复杂运动模式的高度动态主体的光流估计具有挑战性，这可以通过利用更大更多样化的训练数据来缓解；(2) 用于长视野重建的迭代方法在过渡点会产生明显的不一致性；(3) 该方法假设简单的针孔相机模型和干净的 LiDAR 数据，限制了在卷帘快门相机或嘈杂传感器输入下的适用性。更多示例请参阅补充材料。未来的工作将集中在扩展到更大的数据集，开发统一的时间表示以实现无缝的长期重建，并提高对现实世界传感器缺陷的鲁棒性。此外，Flux4D 的显式 3D 表示为世界模型提供了可解释的结构。总体而言，我们相信我们简单且可扩展的设计可以作为社区的基础，从而推动 4D 重建的进一步发展。

疑问：过渡点是什么？

疑问：为什么限制了在卷帘快门相机或嘈杂传感器输入下的适用性？卷帘快门相机或嘈杂传感器和针孔相机模型和干净的 LiDAR 数据的区别是什么？

---

## 6. 结论 (Conclusion)

我们提出了 Flux4D，这是一个可扩展的基于光流的无监督框架，用于通过直接预测 3D 高斯及其运动动力学来重建大规模动态场景。通过仅依靠光度损失并强制执行“尽可能静止”的正则化，Flux4D 有效地分解了动态元素，无需任何监督、预训练模型或基础先验。我们的方法实现了快速重建，能够有效地扩展到大型数据集，并能很好地泛化到未见过的环境。在户外驾驶数据集上的广泛实验表明，在可扩展性、泛化能力和重建质量方面均达到了最先进的性能。我们希望这项工作为大规模高效、无监督的 4D 场景重建铺平道路。

---

## 致谢 (Acknowledgement)
我们衷心感谢匿名审稿人提出的富有洞察力的建议，特别是关于场景流评估、论文框架以及使用单目深度估计模型的额外实验。我们要感谢 Andrei Bârsan 和 Joyce Yang 对初稿的反馈。我们也要感谢 Waabi 团队提供的宝贵帮助和支持。

---

## 参考文献 (References)
*(注：参考文献列表按学术惯例保留英文原文，以便检索)*

[1] Ben Agro, Quinlan Sykora, Sergio Casas, Thomas Gilles, and Raquel Urtasun. Uno: Unsupervised occupancy fields for perception and forecasting. In CVPR, 2024.
[2] David Charatan, Sizhe Lester Li, Andrea Tagliasacchi, and Vincent Sitzmann. pixelsplat: 3d gaussian splats from image pairs for scalable generalizable 3d reconstruction. In CVPR, 2024.
[3] Anpei Chen, Zexiang Xu, Fuqiang Zhao, Xiaoshuai Zhang, Fanbo Xiang, Jingyi Yu, and Hao Su. Mvsnerf: Fast generalizable radiance field reconstruction from multi-view stereo. In ICCV, 2021.
[4] Liang-Chieh Chen, George Papandreou, Florian Schroff, and Hartwig Adam. Rethinking atrous convolution for semantic image segmentation. arXiv preprint arXiv:1706.05587, 2017.
[5] Yuedong Chen, Haofei Xu, Chuanxia Zheng, Bohan Zhuang, Marc Pollefeys, Andreas Geiger, Tat-Jen Cham, and Jianfei Cai. Mvsplat: Efficient 3d gaussian splatting from sparse multi-view images. In ECCV, 2024.
[6] Yun Chen, Matthew Haines, Jingkang Wang, Krzysztof Baron-Lis, Sivabalan Manivasagam, Ze Yang, and Raquel Urtasun. Salf: Sparse local fields for multi-sensor rendering in real-time. arXiv preprint arXiv:2507.18713, 2025.
[7] Yun Chen, Jingkang Wang, Ze Yang, Sivabalan Manivasagam, and Raquel Urtasun. G3R: Gradient guided generalizable reconstruction. In ECCV, 2025.
[8] Yurui Chen, Chun Gu, Junzhe Jiang, Xiatian Zhu, and Li Zhang. Periodic vibration gaussian: Dynamic urban scene reconstruction and real-time rendering. arXiv preprint arXiv:2311.18561, 2023.
[9] Zhe Chen, Yuchen Duan, Wenhai Wang, Junjun He, Tong Lu, Jifeng Dai, and Yu Qiao. Vision transformer adapter for dense predictions. In ICLR, 2023.
[10] Ziyu Chen, Jiawei Yang, Jiahui Huang, Riccardo de Lutio, Janick Martinez Esturo, Boris Ivanovic, Or Litany, Zan Gojcic, Sanja Fidler, Marco Pavone, et al. Omnire: Omni urban scene reconstruction. arXiv preprint arXiv:2408.16760, 2024.
[11] Nathaniel Chodosh, Deva Ramanan, and Simon Lucey. Re-evaluating lidar scene flow for autonomous driving. In WACV, 2024.
[12] Shenyuan Gao, Jiazhi Yang, Li Chen, Kashyap Chitta, Yihang Qiu, Andreas Geiger, Jun Zhang, and Hongyang Li. Vista: A generalizable driving world model with high fidelity and versatile controllability. arXiv preprint arXiv:2405.17398, 2024.
[13] Georg Hess, Carl Lindström, Maryam Fatemi, Christoffer Petersson, and Lennart Svensson. Splatad: Real-time lidar and camera rendering with 3d gaussian splatting for autonomous driving. arXiv preprint arXiv:2411.16816, 2024.
[14] Yicong Hong, Kai Zhang, Jiuxiang Gu, Sai Bi, Yang Zhou, Difan Liu, Feng Liu, Kalyan Sunkavalli, Trung Bui, and Hao Tan. LRM: Large reconstruction model for single image to 3d. In ICLR, 2024.
[15] Anthony Hu, Lloyd Russell, Hudson Yeo, Zak Murez, George Fedoseev, Alex Kendall, Jamie Shotton, and Gianluca Corrado. Gaia-1: A generative world model for autonomous driving. arXiv preprint arXiv:2309.17080, 2023.
[16] Mu Hu, Wei Yin, Chi Zhang, Zhipeng Cai, Xiaoxiao Long, Hao Chen, Kaixuan Wang, Gang Yu, Chunhua Shen, and Shaojie Shen. Metric3d v2: A versatile monocular geometric foundation model for zero-shot metric depth and surface normal estimation. In TPAMI, 2024.
[17] Nan Huang, Xiaobao Wei, Wenzhao Zheng, Pengju An, Ming Lu, Wei Zhan, Masayoshi Tomizuka, Kurt Keutzer, and Shanghang Zhang. S3gaussian: Self-supervised street gaussians for autonomous driving. arXiv preprint arXiv:2405.20323, 2024.
[18] Bernhard Kerbl, Georgios Kopanas, Thomas Leimkühler, and George Drettakis. 3D gaussian splatting for real-time radiance field rendering. In TOG, 2023.
[19] Mustafa Khan, Hamidreza Fazlali, Dhruv Sharma, Tongtong Cao, Dongfeng Bai, Yuan Ren, and Bingbing Liu. Autosplat: Constrained gaussian splatting for autonomous driving scene reconstruction. arXiv preprint arXiv:2407.02598, 2024.
[20] Ishan Khatri, Kyle Vedder, Neehar Peri, Deva Ramanan, and James Hays. I can’t believe it’s not scene flow! In ECCV, 2024.
[21] Tarasha Khurana, Peiyun Hu, David Held, and Deva Ramanan. Point cloud forecasting as a proxy for 4d occupancy forecasting. In CVPR, 2023.
[22] Jaeyeul Kim, Jungwan Woo, Ukcheol Shin, Jean Oh, and Sunghoon Im. Flow4d: Leveraging 4d voxel network for lidar scene flow estimation. In RA-L, 2025.
[23] Alexander Kirillov, Eric Mintun, Nikhila Ravi, Hanzi Mao, Chloe Rolland, Laura Gustafson, Tete Xiao, Spencer Whitehead, Alexander C Berg, Wan-Yen Lo, et al. Segment anything. In ICCV, 2023.
[24] Jinxi Li, Ziyang Song, Siyuan Zhou, and Bo Yang. Freegave: 3d physics learning from dynamic videos by gaussian velocity. In CVPR, 2025.
[25] Siyi Li, Qingwen Zhang, Ishan Khatri, Kyle Vedder, Deva Ramanan, and Neehar Peri. Uniflow: Towards zero-shot lidar scene flow for autonomous vehicles via cross-domain generalization. arXiv preprint arXiv:2511.18254, 2025.
[26] Xiaofan Li, Yifu Zhang, and Xiaoqing Ye. Drivingdiffusion: Layout-guided multi-view driving scenarios video generation with latent diffusion model. In ECCV, 2024.
[27] Xueqian Li, Jhony Kaesemodel Pontes, and Simon Lucey. Neural scene flow prior. In NeurIPS, 2021.
[28] Xueqian Li, Jianqiao Zheng, Francesco Ferroni, Jhony Kaesemodel Pontes, and Simon Lucey. Fast neural scene flow. In CVPR, 2023.
[29] Jeffrey Yunfan Liu, Yun Chen, Ze Yang, Jingkang Wang, Sivabalan Manivasagam, and Raquel Urtasun. Real-time neural rasterization for large scenes. In ICCV, 2023.
[30] Hao Lu, Tianshuo Xu, Wenzhao Zheng, Yunpeng Zhang, Wei Zhan, Dalong Du, Masayoshi Tomizuka, Kurt Keutzer, and Yingcong Chen. Drivingrecon: Large 4d gaussian reconstruction model for autonomous driving. arXiv preprint arXiv:2412.09043, 2024.
[31] Sivabalan Manivasagam, Ioan Andrei Bârsan, Jingkang Wang, Ze Yang, and Raquel Urtasun. Towards zero domain gap: A comprehensive study of realistic LiDAR simulation for autonomy testing. In ICCV, 2023.
[32] Ben Mildenhall, Pratul P Srinivasan, Matthew Tancik, Jonathan T Barron, Ravi Ramamoorthi, and Ren Ng. Nerf: Representing scenes as neural radiance fields for view synthesis. In ECCV, 2020.
[33] Chen Min, Dawei Zhao, Liang Xiao, Jian Zhao, Xinli Xu, Zheng Zhu, Lei Jin, Jianshu Li, Yulan Guo, Junliang Xing, et al. Driveworld: 4d pre-trained scene understanding via world models for autonomous driving. In CVPR, 2024.
[34] Maxime Oquab, Timothée Darcet, Théo Moutakanni, Huy Vo, Marc Szafraniec, Vasil Khalidov, Pierre Fernandez, Daniel Haziza, Francisco Massa, Alaaeldin El-Nouby, et al. Dinov2: Learning robust visual features without supervision. arXiv preprint arXiv:2304.07193, 2023.
[35] Julian Ost, Fahim Mannan, Nils Thuerey, Julian Knodt, and Felix Heide. Neural scene graphs. In CVPR, 2021.
[36] Keunhong Park, Utkarsh Sinha, Jonathan T Barron, Sofien Bouaziz, Dan B Goldman, Steven M Seitz, and Ricardo Martin-Brualla. Nerfies: Deformable neural radiance fields. In ICCV, 2021.
[37] Chensheng Peng, Chengwei Zhang, Yixiao Wang, Chenfeng Xu, Yichen Xie, Wenzhao Zheng, Kurt Keutzer, Masayoshi Tomizuka, and Wei Zhan. Desire-gs: 4d street gaussians for static-dynamic decomposition and surface reconstruction for urban driving scenes. arXiv preprint arXiv:2411.11921, 2024.
[38] Albert Pumarola, Enric Corona, Gerard Pons-Moll, and Francesc Moreno-Noguer. D-nerf: Neural radiance fields for dynamic scenes. In CVPR, 2021.
[39] Ava Pun, Gary Sun, Jingkang Wang, Yun Chen, Ze Yang, Sivabalan Manivasagam, Wei-Chiu Ma, and Raquel Urtasun. Neural lighting simulation for urban scenes. In NeurIPS, 2023.
[40] Jiawei Ren, Cheng Xie, Ashkan Mirzaei, Karsten Kreis, Ziwei Liu, Antonio Torralba, Sanja Fidler, Seung Wook Kim, Huan Ling, et al. L4gm: Large 4d gaussian reconstruction model. In NeurIPS, 2025.
[41] Xuanchi Ren, Yifan Lu, Hanxue Liang, Jay Zhangjie Wu, Huan Ling, Mike Chen, Francis Fidler, Sanja annd Williams, and Jiahui Huang. Scube: Instant large-scale scene reconstruction using voxsplats. In NeurIPS, 2024.
[42] Pei Sun, Henrik Kretzschmar, Xerxes Dotiwalla, Aurelien Chouard, Vijaysai Patnaik, Paul Tsui, James Guo, Yin Zhou, Yuning Chai, Benjamin Caine, Vijay Vasudevan, Wei Han, Jiquan Ngiam, Hang Zhao, Aleksei Timofeev, Scott Ettinger, Maxim Krivokon, Amy Gao, Aditya Joshi, Yu Zhang, Jonathon Shlens, Zhifeng Chen, and Dragomir Anguelov. Scalability in perception for autonomous driving: Waymo open dataset. In CVPR, 2020.
[43] Haotian Tang, Shang Yang, Zhijian Liu, Ke Hong, Zhongming Yu, Xiuyu Li, Guohao Dai, Yu Wang, and Song Han. Torchsparse++: Efficient training and inference framework for sparse convolution on gpus. In MICRO, 2023.
[44] Jiaxiang Tang, Zhaoxi Chen, Xiaokang Chen, Tengfei Wang, Gang Zeng, and Ziwei Liu. Lgm: Large multi-view gaussian model for high-resolution 3d content creation. In ECCV, 2024.
[45] Adam Tonderski, Carl Lindström, Georg Hess, William Ljungbergh, Lennart Svensson, and Christoffer Petersson. NeuRAD: Neural rendering for autonomous driving. In CVPR, 2024.
[46] Haithem Turki, Qi Wu, Xin Kang, Janick Martinez Esturo, Shengyu Huang, Ruilong Li, Zan Gojcic, and Riccardo de Lutio. Simuli: Real-time lidar and camera simulation with unscented transforms. arXiv preprint arXiv:2510.12901, 2025.
[47] Haithem Turki, Jason Y Zhang, Francesco Ferroni, and Deva Ramanan. Suds: Scalable urban dynamic scenes. In CVPR, 2023.
[48] Kyle Vedder, Neehar Peri, Ishan Khatri, Siyi Li, Eric Eaton, Mehmet Kemal Kocamaz, Yue Wang, Zhiding Yu, Deva Ramanan, and Joachim Pehserl. Neural eulerian scene flow fields. In ICLR, 2025.
[49] Jingkang Wang, Sivabalan Manivasagam, Yun Chen, Ze Yang, Ioan Andrei Bârsan, Anqi Joyce Yang, Wei-Chiu Ma, and Raquel Urtasun. CADSim: Robust and scalable in-the-wild 3d reconstruction for controllable sensor simulation. In CoRL, 2022.
[50] Jingkang Wang, Ava Pun, James Tu, Sivabalan Manivasagam, Abbas Sadat, Sergio Casas, Mengye Ren, and Raquel Urtasun. Advsim: Generating safety-critical scenarios for self-driving vehicles. In CVPR, 2021.
[51] Qianqian Wang, Zhicheng Wang, Kyle Genova, Pratul P Srinivasan, Howard Zhou, Jonathan T Barron, Ricardo Martin-Brualla, Noah Snavely, and Thomas Funkhouser. Ibrnet: Learning multi-view image-based rendering. In CVPR, 2021.
[52] Xiaofeng Wang, Zheng Zhu, Guan Huang, Xinze Chen, Jiagang Zhu, and Jiwen Lu. Drivedreamer: Towards real-world-drive world models for autonomous driving. In ECCV, 2024.
[53] Xinyue Wei, Kai Zhang, Sai Bi, Hao Tan, Fujun Luan, Valentin Deschaintre, Kalyan Sunkavalli, Hao Su, and Zexiang Xu. Meshlrm: Large reconstruction model for high-quality meshes. arXiv preprint arXiv:2404.12385, 2024.
[54] Benjamin Wilson, William Qi, Tanmay Agarwal, John Lambert, Jagjeet Singh, Siddhesh Khandelwal, Bowen Pan, Ratnesh Kumar, Andrew Hartnett, Jhony Kaesemodel Pontes, et al. Argoverse 2: Next generation datasets for self-driving perception and forecasting. arXiv preprint arXiv:2301.00493, 2023.
[55] Guanjun Wu, Taoran Yi, Jiemin Fang, Lingxi Xie, Xiaopeng Zhang, Wei Wei, Wenyu Liu, Qi Tian, and Xinggang Wang. 4d gaussian splatting for real-time dynamic scene rendering. In CVPR, 2024.
[56] Tianhao Wu, Fangcheng Zhong, Andrea Tagliasacchi, Forrester Cole, and Cengiz Oztireli. Dˆ2nerf: Self-supervised decoupling of dynamic and static objects from a monocular video. In NeurIPS, 2022.
[57] Pengchuan Xiao, Zhenlei Shao, Steven Hao, Zishuo Zhang, Xiaolin Chai, Judy Jiao, Zesong Li, Jian Wu, Kai Sun, Kun Jiang, et al. Pandaset: Advanced sensor suite dataset for autonomous driving. In ITSC, 2021.
[58] Haofei Xu, Songyou Peng, Fangjinhua Wang, Hermann Blum, Daniel Barath, Andreas Geiger, and Marc Pollefeys. Depthsplat: Connecting gaussian splatting and depth. arXiv preprint arXiv:2410.13862, 2024.
[59] Yunzhi Yan, Haotong Lin, Chenxu Zhou, Weijie Wang, Haiyang Sun, Kun Zhan, Xianpeng Lang, Xiaowei Zhou, and Sida Peng. Street gaussians for modeling dynamic urban scenes. In ECCV, 2024.
[60] Jiawei Yang, Jiahui Huang, Yuxiao Chen, Yan Wang, Boyi Li, Yurong You, Maximilian Igl, Apoorva Sharma, Peter Karkus, Danfei Xu, Boris Ivanovic, Yue Wang, and Marco Pavone. Storm: Spatio-temporal reconstruction model for large-scale outdoor scenes. arXiv preprint arXiv:2501.00602, 2025.
[61] Jiawei Yang, Boris Ivanovic, Or Litany, Xinshuo Weng, Seung Wook Kim, Boyi Li, Tong Che, Danfei Xu, Sanja Fidler, Marco Pavone, and Yue Wang. Emernerf: Emergent spatial-temporal scene decomposition via self-supervision. arXiv preprint arXiv:2311.02077, 2023.
[62] Ze Yang, Yun Chen, Jingkang Wang, Sivabalan Manivasagam, Wei-Chiu Ma, Anqi Joyce Yang, and Raquel Urtasun. Unisim: A neural closed-loop sensor simulator. In CVPR, 2023.
[63] Ze Yang, Sivabalan Manivasagam, Yun Chen, Jingkang Wang, Rui Hu, and Raquel Urtasun. Reconstructing objects in-the-wild for realistic sensor simulation. In ICRA, 2023.
[64] Ze Yang, Jingkang Wang, Haowei Zhang, Sivabalan Manivasagam, Yun Chen, and Raquel Urtasun. Genassets: Generating in-the-wild 3d assets in latent space. In CVPR, 2025.
[65] Zetong Yang, Li Chen, Yanan Sun, and Hongyang Li. Visual point cloud forecasting enables scalable autonomous driving. In CVPR, 2024.
[66] Ziyi Yang, Xinyu Gao, Wen Zhou, Shaohui Jiao, Yuqing Zhang, and Xiaogang Jin. Deformable 3d gaussians for high-fidelity monocular dynamic scene reconstruction. In CVPR, 2024.
[67] Yuanwen Yue, Anurag Das, Francis Engelmann, Siyu Tang, and Jan Eric Lenssen. Improving 2D Feature Representations by 3D-Aware Fine-Tuning. In ECCV, 2024.
[68] Haiming Zhang, Wending Zhou, Yiyao Zhu, Xu Yan, Jiantao Gao, Dongfeng Bai, Yingjie Cai, Bingbing Liu, Shuguang Cui, and Zhen Li. Visionpad: A vision-centric pre-training paradigm for autonomous driving. arXiv preprint arXiv:2411.14716, 2024.
[69] Kai Zhang, Sai Bi, Hao Tan, Yuanbo Xiangli, Nanxuan Zhao, Kalyan Sunkavalli, and Zexiang Xu. GS-LRM: Large reconstruction model for 3D gaussian splatting. In ECCV, 2025.
[70] Lunjun Zhang, Yuwen Xiong, Ze Yang, Sergio Casas, Rui Hu, and Raquel Urtasun. Learning unsupervised world models for autonomous driving via discrete diffusion. In ICLR, 2024.
[71] Wenzhao Zheng, Weiliang Chen, Yuanhui Huang, Borui Zhang, Yueqi Duan, and Jiwen Lu. Occworld: Learning a 3d occupancy world model for autonomous driving. In ECCV, 2024.
[72] Xiaoyu Zhou, Zhiwei Lin, Xiaojun Shan, Yongtao Wang, Deqing Sun, and Ming-Hsuan Yang. DrivingGaussian: Composite gaussian splatting for surrounding dynamic autonomous driving scenes. In CVPR, 2024.
[73] Chen Ziwen, Hao Tan, Kai Zhang, Sai Bi, Fujun Luan, Yicong Hong, Li Fuxin, and Zexiang Xu. Long-lrm: Long-sequence large reconstruction model for wide-coverage gaussian splats. arXiv preprint arXiv:2410.12781, 2024.