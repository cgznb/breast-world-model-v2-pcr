# 实验设计与验收顺序

## 先验证可用性，而不是直接开始十万步

用 smoke 验证解释器和完整代码链。然后使用真实 manifest 执行 `audit --scan-arrays`，检查患者划分、phase order、latent shapes、时间可用性、图像监督数量、triplet rejection reasons。

首个真实实验建议保留 cache-safe 模式，A/B 各使用一个新的小预算调试配置；不要在 resume 过程中修改总步数。检查 same-patient source/target shape、三个 phase 的原 VQ 重建、每项 loss 非零的有效监督数量、训练/验证轨迹和非有限值。确认 decoder input 必须先反标准化，匹配原 VQ。

有真实训练 sidecars 后再比较 kinetics、segmentation、biomarker/pCR，而不是将 `task_support=0` 的项目计入“已使用医学约束”。teacher 的默认接口可以跑通，但没有真实导出的合适 pretrained 特征时不得在论文写“使用某基础模型”。

## 主要对照

最重要对照是**完全保留的原仓库三相 SymmFlow**，固定相同 patients、pairs、VQ、原图处理、训练预算、采样步数和选择协议。包内 `ablations/velocity_only.yaml` 是新版框架内去掉额外生成损失的控制，不是上游 exact baseline：它仍可能使用 A 阶段预训练的 clinical tokenizer，不能与原始代码混为一谈。

建议用以下最小实验矩阵检验方法假设：

| 实验 | 表征 | 生成约束 | 目的 |
|---|---|---|---|
| 原始上游 | 原 frozen VQ | 原两分支 velocity MSE | 真正原始 baseline |
| 新版、无 JEPA | phase-aware 多尺度 state | 同样生成预算 | 验证 masked predictive 学习作用 |
| 新版、无 future predictor | phase-aware + JEPA | 不给 predictive prior | 分开 future loss/prior 的贡献 |
| 新版、无 state constraints | 不加 anatomy/separation/latent-delta/医学 heads | 固定 dynamics | 检验表征约束不是装饰 |
| 新版、无 local map | 完整 A | FM + semantic endpoint | 检验数值一致性是否有额外作用 |
| Cache-safe 完整 | 完整 A | B | 主体方法在现有缓存上的效果 |
| Registered-full | 完整 A | B + C | 在同一合格子集检验真实 open-loop rollout |
| Registered-full 去掉 C | 与上一行相同 | B | 不改变合格患者子集的 composition 消融 |

不能用更多数据、更大 backbone、更多 optimizer updates 与额外 NFE 后将增益全部归因于一个新 loss。对 A 的预训练成本和 C 的多次 ODE 调用单独计时。至少比较固定 optimizer update budget 和固定 wall-clock/计算预算两种条件；生产参数量已给出，full-size GPU 显存需要实际测量。

## 什么结果才支持世界模型方向

一跳：分别报告 T0→T1、T0→T2、T0→T3 和其他合法 pair，不仅报告混在一起的总体平均。必须包含 copy-source，对重建本身先报告 target VQ reconstruction 上限/误差，以分清 codec 瓶颈和 dynamics 瓶颈。

多跳：T0→预测 T1→预测 T2 和 direct T0→T2 比较；第二步输入必须为预测 T1。另设真实 T1 输入的 observation-update 版本作为上界/不同任务，不混入 open-loop 主结果。预测时段计划必须在 T0 已知，或明确写成外部指定场景；不能使用后来观测后才决定的实际方案，却称其为 baseline forecasting。

临床读出：有足够标注时，报告真实肿瘤体积/FTV 变化、分割、三相 signal difference 的误差、pCR 区分度和 calibration，按亚型、方案、响应/非响应分层。论文不能仅用模型自己训练的 semantic encoder 给自己的输出打分。包内 pCR 是 source-state 辅助分类器；如将 generated MRI 接入原 Pillar downstream，要保持原 classifier 的独立训练/选择协议，并明确该扩展不在当前自动评价入口内。

不确定性：本包提供 ensemble sampling、diversity 和 feature-space energy score。正式研究要增加可信医学 biomarker 的 coverage、校准与错误相关性；K=2 仅是训练成本起点，不能用两个样本宣称充分校准。不得用真实 target 挑选最像它的随机样本再计算“预测精度”。

跨域：scanner/site、缺失相位、时间间隔范围、患者亚型可能影响结果。单个 I-SPY2 setting 的改善不自动证明 general world-model representation。需要在数据允许的条件下做独立中心/数据集外部验证。

## 统计协议

患者是独立分析和 bootstrap 的基本单元，不以同一患者多个 all-pairs 样本当独立病例扩大显著性。对各实验用一致划分及不少于三个合理随机种子，报告均值、区间、失败运行与计算开销。少数类别 pCR AUROC 不稳定时，明确样本数和单类无法计算的情况；不能将 null 替换为 0.5 后隐瞒。

配准 QA 淘汰必须在各对照保持一致并报告排除原因；不得只为新方法挑选容易配准、响应明显的患者。外部权重/数据协议、亚组标签、训练分布信息都需核验。

## 当前验收证据的正确用法

`reports/pytest.xml` 是代码测试记录。`reports/smoke_report.json` 是完全合成数据运行证据。`reports/production_parameter_counts.json` 是真实实例化的参数统计。它们支持“代码链可执行和部分数值/契约正确”，不支持“已在 I-SPY2 提升 pCR AUC”、“已达到 CVPR 水平”或“可指导临床”。
