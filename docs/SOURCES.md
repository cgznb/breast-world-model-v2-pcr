# 实际查阅的公开代码与复用边界

下表记录本次实现检查的文件和实际使用方式。Git blob 是文件版本，不是整个仓库的 commit。没有把整套作者代码或 pretrained weights 冒充打包进来。具体 source metadata 见 `SOURCE_AUDIT.json`。

| 来源 | 核对文件 | 本版具体位置 | 复用类型与边界 |
|---|---|---|---|
| cgznb/symm-fm | `three_phase_symmflow.py`, `three_phase_all_pairs_data.py`, `first_post_world_data.py`, `vqgan.py`, `flow/path.py`, `training/datasets.py` | `data.py`, `flow.py`, `codec.py` | 原三相/条件/latent 契约适配；codec 为具名来源的推理解码子集 |
| I-JEPA, facebookresearch/ijepa | `src/train.py`，teacher/context/loss 部分 | `models.py`, `encoder.py`, `losses.py` | EMA clean teacher、masked representation prediction；新的 3D latent 适配，不是原图像预训练复现 |
| V-JEPA 2, facebookresearch/vjepa2 | `src/models/predictor.py` | `layers.TokenPredictor`, masked/future predictors | 多层 Transformer predictor、learned mask query、位置编码的设计参考；没有加载机器人/视频权重 |
| Video Swin / torchvision | `torchvision/models/video/swin_transformer.py` | `layers.py` | 3D window/shift/relative bias/mask 算法适配，另加 padding key mask，保留 BSD notice |
| ConvNeXt | `models/convnext.py` 的 Block | `layers.ConvNeXt3D` | k7 depthwise、channels-last LN、4×MLP、layer scale 的 3D 适配 |
| REPA, sihyun-yu/REPA | `loss.py` | `losses.py`, `teachers.py` | clean-feature alignment 启发；本版是 endpoint-semantic/teacher alignment，不冒充原 REPA hidden-state 实验 |
| Consistency Flow Matching | `losses.py` 的 consistency function | `flow.local_teacher_map`, `losses.pair_flow_loss` | 阅读多段/端点/velocity 一致性，避免直接错误移植；本版使用 EMA Heun local map，非原 CFM 全复现 |
| VICReg | `main_vicreg.py` 的 variance/covariance | `losses.variance_covariance`, `models.moment_queue` | 方差/协方差正则，微批次队列为本版适配，不等价原跨 GPU batch |
| MONAI 1.5.1 | `monai/networks/nets/diffusion_model_unet.py` | `velocity.MonaiVelocityUNet` | 真正调用该库模型并核对 API；本环境未安装 MONAI，该后端未执行 |

可核对的原始资料：

- I-JEPA： https://arxiv.org/abs/2301.08243 ，代码 https://github.com/facebookresearch/ijepa
- V-JEPA 2： https://arxiv.org/abs/2506.09985 ，代码 https://github.com/facebookresearch/vjepa2
- Video Swin： https://arxiv.org/abs/2106.13230 ，适配参考 https://github.com/pytorch/vision/blob/main/torchvision/models/video/swin_transformer.py
- ConvNeXt： https://arxiv.org/abs/2201.03545 ，代码 https://github.com/facebookresearch/ConvNeXt
- REPA： https://arxiv.org/abs/2410.06940 ，代码 https://github.com/sihyun-yu/REPA
- Consistency Flow Matching： https://arxiv.org/abs/2407.02398 ，代码 https://github.com/YangLing0818/consistency_flow_matching
- VICReg： https://arxiv.org/abs/2105.04906 ，代码 https://github.com/facebookresearch/vicreg
- MONAI： https://github.com/Project-MONAI/MONAI/tree/1.5.1
- 原项目： https://github.com/cgznb/symm-fm/tree/92265d3b1749ae3b686f2089843c49da129fd4d2

Flow Map Matching 等数值流工作可用于理解 flow-map composition，但不能直接用来证明医学临床时间的 Markov semigroup。这里的临床 composition 是带有准入条件、需要消融验证的假设。

本次没有自动加载或实现完整 VoCo/SwinUNETR pretrained backbone，也没有完整 3D DiT 替换。外部 teacher 是明确的 TorchScript 接口；需要使用者提供适合 MRI 的可信 checkpoint、导出层、预处理和训练数据独立性证据。不能在论文中仅凭接口存在就声称已经使用某预训练模型。
