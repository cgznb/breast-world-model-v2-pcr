# 详细修改方案：Predictive DCE State + Guarded Stochastic Symmetric Dynamics

版本 0.2.0。研究实现，不代表临床有效性已经建立，也不保证发表结果。文中“创新方向”指可检验的研究假设，不是已完成优先权检索的原创性结论。公开代码核对范围见 `SOURCES.md`。

## 一、先纠正上一轮建议中的三个问题

第一，不能将新学出的 24-channel semantic state 不经适配直接送进原 frozen VQ decoder。形状一样不等于表示空间一样。原 decoder 学的是某一 VQ encoder 的坐标分布，不认识新分解出的 anatomy/disease channels。因此新版保留原 continuous VQ latent 作为生成变量，另外训练深层 semantic encoder。病人状态负责条件、预测和约束，旧 VQ 坐标负责图像解码。没有暗中训练一个换坐标后仍被称为“兼容旧 VQ”的模型。

第二，随机模型的一次 T0→T2 采样和一次 T0→T1→T2 采样通常不应逐元素相等。直接对两次独立噪声生成的样本做 MSE，会鼓励结果趋于相同，削弱建模不确定性的目的。新版用**每患者、每场景的样本分布 MMD**匹配 direct/composed rollout，再用 energy score 将 composed 分布锚定到真实终点。没有 best-of-K 选择，也没有让每个随机样本都复制单一真实随访。

第三，线性训练插值的“样本条件速度为常量”，不意味着实际学到的 marginal velocity 在任意两个时间点必须一致。新版没有简单加入 `v(t1)==v(t2)`。采用 EMA teacher 在短 flow-time 区间作 Heun 积分，学生匹配短区间 flow map；这是带有步长误差/偏差的数值蒸馏正则，不是“真实医学变化速度必须常量”的假设。它也不是对原 Consistency-FM 全部多段训练细节的复现。

## 二、现有代码的确切边界与改动方式

核对的上游：`cgznb/symm-fm`，提交 `92265d3b1749ae3b686f2089843c49da129fd4d2`。

`three_phase_symmflow.py` 的三相为 `pre_aqc0 / first_post_aqc1 / metadata_late`，不是 T0/T1/T2。各相 `[1,96,256,256]` 独立经过同一个 frozen single-channel VQ 的 `encode_continuous`，得到各 `[8,24,64,64]`，拼为 `[24,24,64,64]`。SymmFlow 的两分支拼接为 48 channels。

新版采取增量工作流，旧三相入口和 baseline 保留。新入口 `world.py` 或安装后的 `run_world_v2.py`，新模块 `src/symm_world/`。旧 checkpoint 必须继续用旧入口；新网络不能拿旧 optimizer state 假装 resume。VQ checkpoint 可经 `codec.py` 的兼容读取器加载，不包含旧训练系统的 Lightning/discriminator 依赖。

原生缓存只给 train 保存 latent、对 val 另存 image references 的情况已经在适配器中考虑。未配准、不同视图裁剪、未来中心定位不能被“改配置开关”自动变成合法纵向坐标。

## 三、完整网络结构

### 3.1 两层状态，不破坏图像生成坐标

令原 frozen VQ 的输出为 `z_t ∈ R^{24×D×H×W}`。用训练集按 channel 标准化，但保留均值和标准差以便解码。新 encoder 输出：

- `dense_t ∈ R^{32×192}`：生产配置的 2×4×4 空间 tokens。
- `anatomy_t ∈ R^{4×192}`：学习到的相对稳定信息查询 tokens。
- `disease_t ∈ R^{8×192}`：用于响应预测的查询 tokens。

这是可解释性导向的参数化，不是已证明的病人生物学解耦。尤其不能声称 anatomy 完全不随治疗、体位或扫描协议变化；稳定性约束仅作用于经 QA 认定可比较的 pair。

### 3.2 Phase-aware 多尺度 Encoder：逐层

输入 `[B,24,24,64,64]`，先拆成 `[B×3,8,24,64,64]`。共享 `Conv3d(8,48,k=3,stride=2)`，得到每相 `[48,12,32,32]`；再经过两个 ConvNeXt3D block。每个 block 包含 `depthwise Conv3d(k=7)`、channels-last LayerNorm、4 倍扩展 MLP、GELU、projection、layer scale 和 residual。

将同一空间位置的三个 phase feature 组成长度为 3 的序列，加入可学习 phase embedding，通过 **2 个 pre-norm TransformerEncoder block**，每 block 为 multi-head attention + 4×FFN。三个 phase 的顺序具有明确语义，不进行任意置换不变约束。

并行计算三个 8-channel 特征差：early−pre、late−pre、late−early。先还原各相原 latent 坐标，再使用共享尺度，而不是直接减去分别 z-score 后且尺度不同的三相。它们是**编码特征差**，不能称作 MRI enhancement ratio。差分分支经过 stride-2 3D convolution 和 ConvNeXt3D，然后与 phase fusion 相加。

后续为三个真正的 3D shifted-window Transformer level：

| Level | 通道 | Block 数 | Heads | 空间尺寸，标准输入时 |
|---|---:|---:|---:|---|
| 0 | 96 | 2 | 3 | 12×32×32 |
| 1 | 192 | 2 | 6 | 6×16×16 |
| 2 | 384 | 4 | 12 | 3×8×8 |

window 为 2×4×4，交替非 shifted / shifted，包含 3D relative position bias、cyclic boundary mask 和 padding key mask。层间采用八个子体素拼接的 3D patch merging，不是将深度维当作普通 batch。

各层 projection 到 192 channels，再 adaptive pool 至 2×4×4。将三层的同位置 feature 拼接，经过多层 fusion 和 LayerNorm，加入固定 3D positional embedding。两个不同的 learned-query attention pooling 头产生 anatomy/disease tokens。

生产配置实际实例化统计（类别词表仅使用示例词表计数）：semantic encoder 10,601,958 参数；完整 representation system 可训练 17,272,672 参数。native velocity U-Net 63,497,184 参数；world stage 可训练 64,387,296 参数，包括条件 adapter。精确值和词表条件见 `reports/production_parameter_counts.json`。这些数字不是全尺寸显存或医学精度测试结果。

### 3.3 Masked JEPA Predictor

先在 **cached latent 输入、任何新卷积与 phase mixing 之前**遮蔽 coarse spatial blocks，再以一定概率遮蔽整个 DCE phase。未遮蔽的真实 future visit 不会输入此 context branch。EMA target encoder 读取同一个 visit 的干净 latent，输出 stop-gradient target。

`MaskedStatePredictor` 使用 learned mask queries + 3D position、**4 层 TransformerDecoder**对 context dense tokens 做预测，只在被空间 mask 的位置计算 smooth-L1。

这是 latent-space JEPA-style 适配，不是原 I-JEPA 图像输入/多块采样实现的逐项复现。原 frozen VQ 的相邻 latent 本身可能有重叠 receptive field，因此不声称原始 MRI 像素在信息论意义上完全独立遮蔽。这个限制在本缓存版本中保留。

### 3.4 临床条件与 Residual State Predictor

临床字段只允许固定白名单：treatment_arm、HR/HER2、mammaprint、age、stage_i/j、核验后的 delta_days 和显式治疗计划。

缺失类别、未知类别、缺失数值分别表示。没有核验的脱敏日期不会进入时间数值 token。每个 treatment segment 包含 drug、start、end、dose、known_at_source；药物类别 embedding 与区间/剂量 MLP 相加，然后和基础临床 tokens 经过两层 Transformer。

未来状态预测器不是简单 Linear：使用 **4 层 TransformerDecoder**，以当前 disease tokens 为 queries，以 anatomy + disease + clinical tokens 为 memory：

`pred_disease_j = disease_i + alpha × Predictor(disease_i, anatomy_i, clinical_ij)`。

这里 residual 是**语义状态 residual**，不是将原 VQ 全部 channel 武断分为静态/动态两半。生成 MRI 的 SymmFlow 仍预测完整的原 VQ latent 分布。这个设计维持双向采样与 frozen decoder 契约。

### 3.5 World Dynamics

保留原两分支 SymmFlow path。条件不只有临床字段：source anatomy tokens、source disease tokens、source-derived predictive tokens 共同经过两层 trainable context adapter，输入 velocity backbone。正向预测的 predictive tokens 只由 source 和已知计划计算，绝不由真实 future encoder 输出充当条件。

提供两个明确后端。`monai` 真正实例化 MONAI 1.5.1 的 `DiffusionModelUNet`；生产宽度 128/256/384、每尺度两个 ResBlock、3D 多尺度、条件 cross-attention。`native` 是包内完整的三尺度 3D 条件 U-Net，包含 FiLM residual blocks、bottleneck shifted-window self-attention、clinical/patient cross-attention、skip connections。不是为了通过测试替换成的 dummy 网络，两者也不宣称完全相同。

零初始化输出层可能使第一个 update 的梯度主要落在输出层；从后续 update 检查 upstream gradients。这不是 EMA 或 stop-gradient 导致训练完全断开的证据。

## 四、A 阶段：让 representation 对患者状态有用

A 阶段 train：semantic encoder、clinical encoder、masked predictor、future predictor 和 heads；EMA target encoder 不接受反传。原 VQ 不训练，不产生旧缓存失效。

核心目标为：

`L_A = L_coarse-recon + L_JEPA + 0.25 L_future + 0.1 L_var + 0.01 L_cov + 0.02 L_anatomy + 0.01 L_separation + 0.1 L_latent-delta + available auxiliary losses`。

权重是可调整的初始研究配置，不是经验最优值。医学辅助头并非用来“增加网络深度”，它们为表征提供训练信号；主干和 predictors 已经是多层结构。

Coarse reconstruction 预测空间池化后的 24-channel VQ representation，防止语义头只学常量，并保留一些空间信息。JEPA 预测 masked clean semantic tokens。Future loss 从 source disease state 预测 EMA future disease state，是表征预训练的辅助回归，不将它冒充完整未来分布生成。

Variance/covariance 参考 VICReg。dense tokens 提供局部统计；微批次为 1 时另使用一个 64 个历史患者 summary 的 detached queue 增补患者层面统计。历史队列可能包含同一患者多次采样、也存在 encoder drift，因此它不是独立患者大 batch 的严格等价替代。报告 representation variance 和下游任务，不能仅凭正则认定“不会 collapse”。

Anatomy consistency 只对 `anatomy_comparable=true` 的 pair 使用 pooled feature；不比较未配准的逐 voxel anatomy，不强行保持肿瘤区域不变。Separation 使用两个 pooled summary 的 cross-covariance；零 covariance 不证明独立或因果可辨识。

有真实同 visit 图像时，kinetic head 预测三个信号差，要求 documented shared scalar normalization 和有效三相覆盖 mask。有可靠肿瘤 mask 时使用 soft BCE + Dice；全零肿瘤是有效标注，缺失 mask 则不计算 loss。四个 biomarker targets 由你的数据契约明确定义与标准化，不把任意四个值冒充 FTV/生物标志物。pCR head 只读取 source state 和 source-available conditions，pCR 标签仅在训练监督中使用。

可选 external teacher alignment 对 `external_projection(dense)` 与离线导出的可信 teacher feature 做 cosine loss。不包含预训练权重，也不声称内部自监督 encoder 是现成医学 foundation model。

## 五、B 阶段：双向随机 SymmFlow 与非重复约束

冻结 A 阶段选出的表征模块和辅助 heads。训练 velocity backbone 和 context adapter；另维护 world EMA teacher。

设较早 latent 为 `E`，较晚 latent 为 `L`：

`x_tau=(1−tau)epsilon_x + tau L`

`y_tau=(1−tau)E + tau epsilon_y`

`v*=(L−epsilon_x, epsilon_y−E)`。

一次网络前向输出两分支 velocity；主 loss 为两个分支 MSE 之和。部分 batch 使用 clean earlier 做条件训练 forward，其余使用 clean later 做条件训练 retrodiction。direction token 区分任务，训练 input/target 的角色清楚分离。

由同一次 velocity 预测可以估计：

`L_hat=x_tau+(1−tau)v_x`

`E_hat=y_tau−tau v_y`。

这些是单次 denoised endpoint estimates，不是完整 ODE sample。直接加 `||L_hat−L||²` 在代数上等于 `(1−tau)²||v_x−v_x*||²`；因此本实现没有把它包装成独立 innovation。真正新增的是**非线性的冻结语义 encoder / 已训练辅助头 / 可选 VQ decoder**上的误差。

`L_B = L_velocity + 0.05 L_semantic-endpoint + 0.02 L_local-map + available endpoint readout losses + optional decoded-kinetics`。

Semantic endpoint 与 REPA 的关联仅在于使用 clean representation 约束生成学习。原 REPA 对齐生成网络内部 hidden features；这里对齐的是单步 endpoint 经冻结 encoder 的表征，**不叫原样 REPA**。

Local map：EMA teacher 从当前 joint state 在 `[tau,tau+delta]` 用 Heun 做短步积分；学生 Euler map 匹配 teacher map，并以 `delta` 归一化。默认 delta=0.05，接近边界的极小区间跳过，B 前 2000 optimizer steps 不加此类辅助项，其后每 4 step 使用。这样减少早期不可靠 teacher 的影响和开销，但并非已经证明最优。

Decoded kinetics：可选每次截取一个 latent patch，经 frozen quantizer 的 straight-through derivative 和 frozen decoder 解码，比较三相信号差。decoder parameters/codebook 均不更新，梯度对 predicted latent 保留。比较对象是匹配 VQ 的 target reconstruction，明确称 codec reconstruction proxy；不等于原始患者真实 enhancement curves。crop 边界被忽略以减小局部 padding 影响。

冻结模型参数不等于 `torch.no_grad()`：生成端点经过 frozen semantic encoder、second-hop 经过 frozen encoder、预测 latent 经过 frozen decoder时，都必须保留输入梯度。测试覆盖了这些路径。

## 六、C 阶段：带有准入条件的随机 Clinical-Time Composition

临床时间不是 FM 的 tau。Flow Map Matching 等方法对数值生成流的 flow-map structure 的讨论，不能直接证明乳腺治疗过程满足 Markov semigroup。

本实现将“足够状态 + 相同已声明干预场景下的组合一致性”视为**可消融的 inductive bias**。若观察状态不满足 Markov 性、存在未观测共病或临床决策反馈，这个约束可能产生偏差，应该降低权重或关闭，而不是宣称自然成立。

每个候选 `(i,j,k)` 都需检查：相同患者及 split；六个相关 source/target view 的 grid ID 和实际 geometry 均兼容；输出定位在 initial source 已可确定；相应 pair 注册已核验；真实临床间隔核验且可相加；基础临床信息相容；治疗分段计划可组合；三个 transition 的 scenario_id 相同；完整计划在初始 source 已经知道。**中间时点后来才决定的实际治疗，不可因为在中间 visit 已知就偷渡进 baseline rollout**。

注意：保留 latent shape 相同不足以满足上述条件。上游 `target_center_source_extent...` 可以使同一 visit 存在多套裁剪坐标；源码中的 source-only 文件读取 audit 也不能自行证明预处理没有使用未来定位。本包不会将那些配对直接当作 registered triplet。

对每个合格 triplet，独立抽取 K≥2 组噪声：

`z_j^(k) = F_theta(z_i,c_ij,epsilon_1^k)`

`z_k,composed^(k) = F_theta(z_j^(k),c_jk,epsilon_2^k)`

`z_k,direct^(k) = F_EMA(z_i,c_ik,epsilon_3^k)`。

真实中间 MRI 不读取、不输入第二步。真实终点仅作监督 anchor。Direct teacher 分支 stop-gradient，composed 两步都接受 gradient，frozen encoder 也保留对合成输入的导数。

分布在冻结 disease feature 空间中比较。默认多 bandwidth 的 biased RBF MMD：

`MMD² = mean k(X,X)+mean k(Y,Y)−2mean k(X,Y)`。

按**每患者的 K 维样本轴**计算，不把其他患者的真实/生成样本混在一起代替同患者条件分布。K 很小的 MMD 估计方差/偏差仍明显；默认 K=2 是可运行的起点，不是充分的不确定性评估。

终点 anchor 使用 energy score：

`ES = (1/K) Σ||X_k−y|| − [1/(2K(K−1))]Σ_{k≠l}||X_k−X_l||`。

第二项保留样本差异的奖励，不强迫所有 samples 等于单个患者观察。该 proper score 的保证在指定特征空间和分布假设下成立；若 learned feature 不足，不能据此证明整个 MRI 分布正确。

`L_C = L_B + 0.1 L_distribution-MMD + 0.1 L_energy-anchor`。

C 从 B 最佳 checkpoint 对应的 EMA world 权重初始化，不冻结 velocity。默认 every=4、K=2、每 transition 4 Heun steps。每个 triplet 额外 student NFE 为 `2 legs × 2 Heun evaluations × steps × K`，还需要 direct teacher evaluations；**不宣称 C 只多一次前向**。默认情况下，额外 student NFE=32，teacher NFE=16（不含 context encoder 和 representation reads），只在选中的更新执行。需要记录真实显存和 wall-clock budget。

对于原生未配准缓存，B 阶段首先是标准化 ROI 表观/表征预测，而不是已证明处于固定解剖坐标下的肿瘤生长预测。输出 NPZ 的数组形状匹配 source latent，但独立推理不能从裸 NPY 恢复真实 physical origin/direction；需要另行核验源几何后才能写入 NIfTI。

所有这些是 associational scenario modeling；本包不估计真实因果 treatment effect、不自动搜索最优临床治疗，也不提供治疗建议。

## 七、运行、保存和评价保证

每个阶段分别保存完整 student、EMA、optimizer、scheduler、torch RNG、metadata、step 和 best score。数据、配置和资产签名必须一致才允许 resume。默认患者→transition→pair 的采样方式，避免随访多的患者因 all-pairs 产生大量样本而主导训练。

标准化和临床词表仅用 train fit。病例 split 相交直接报错。训练缺失 pCR/辅助标签不默认置为负类；checkpoint 记录实际出现过有效监督的任务，未训练的 pCR head 不在公开 sample 输出中产生概率。

推理入口只接受 source latent、source-available conditions、checkpoint、随机种子；需要重建 MRI 时另读匹配 VQ。没有 target 参数。评价程序先执行 source-only generation，之后才读真实 target 做评分。

目前内置评价报告 standardized-latent MAE/RMSE、copy-source 对照、样本多样性、semantic energy score、每相 latent MAE、患者 bootstrap interval，以及已训练 pCR source head 的分 landmark 评价。这**不是** MRI PSNR/SSIM、真实 FTV 误差、临床 calibration 或生存分析的完整替代。pCR head 当前评估 source state，不是对生成未来 state 联合训练的 prognostic head。

## 八、对应上一轮创新点的落实状态

| 上一轮方向 | 本版落实 | 边界 |
|---|---|---|
| Joint phase-aware encoder | 多层 ConvNeXt3D + cross-phase Transformer + 3D Swin | 作用在原 VQ cache 上，不重新训练原 VQ encoder |
| DCE kinetic encoding | 显式 latent 差分；真实图像/codec proxy sidecar、kinetic head | latent 差分不冒充生理信号，必须区分 measured/proxy |
| JEPA 预测编码 | EMA target + block/phase masking + 4 层 predictor | 是 3D latent 适配，不宣称原 I-JEPA 原样复现 |
| Future-predictive state | 4 层临床条件 residual disease predictor | 表征辅助回归不等于生成分布 |
| Static/dynamic 解耦 | anatomy/disease 查询头 + 弱约束 + 去相关 | 不硬复制 anatomy，不声称可辨识生物解耦 |
| Residual treatment dynamics | source-conditioned residual semantic prior | 原 VQ FM 不换成未经推导的 residual endpoint path |
| Velocity consistency | EMA local Heun flow-map regularizer | 替换错误的任意两时刻速度相等，不是原 CFM 全复现 |
| Clinical composition | gated stochastic MMD + energy score + 两步梯度 | 需要正确配准、时间和初始已知计划；默认缓存版关闭 |
| pCR auxiliary | source-only supervised classifier | 真实标签仅训练用，不需要推理病理输入 |
| 先进 pretrained encoder | 可信本地 dense teacher 的实际导出/对齐接口 | 没有捏造或打包 pretrained weights，没有自动 VoCo 权重适配 |
| 更换 velocity backbone | 原 MONAI option + 完整 native 3D U-Net | 未实现/声称 3D DiT pretrained 迁移 |
| 强行正反 cycle | 不实施 | retrodiction 不等于生物过程可逆 |

## 九、发表前需要解决的事实，而不是再堆模块

应证明 phase mixing 与 predictive representation 对真实病人相关指标有贡献，而不是只提升 learned-space 自评误差；应证明 multi-step consistency 在严格独立患者测试上改善 rollout，且不损害 diversity/calibration；应与 untouched 原仓库和同预算对照比较。这里提供的是可执行研究系统和可复现实验起点，不是已经达到某会议接收要求的性能证据。
