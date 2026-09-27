# Breast World Model V2 + pCR

三相乳腺 MRI 世界模型 V2，以及完整的外部 PCR 预测代码。
本次快照整理于 2026-09-27，包含完整 V2 和简化 V2 的配置。

## 代码组成

| 目录 | 内容 |
| --- | --- |
| `src/symm_world/` | 状态编码器、未来特征预测器、SymmFlow、VQ 接口、训练与推理 |
| `configs/registered_roi32_5090.yaml` | 完整 V2 的实际 ROI32 配置 |
| `configs/ablations/registered_roi32_5090_simple_a.yaml` | 简化 V2 配置 |
| `pcr/` | 冻结 Pillar、TDN、临床先验、训练、预测与验证代码 |
| `vendor/mewm-ispy2/` | 原 ROI32 数据、预处理和 VQ 依赖源码与许可 |
| `tests/` | 生成模型及输入隔离、训练恢复等测试 |

A 阶段从 `[24,8,32,32]` 三相 VQ latent 编码得到 anatomy/disease tokens，
以临床和时间条件预测未来 disease tokens。B 阶段冻结 A，使用条件 3D U-Net 做双分支 SymmFlow。
完整 V2 的 A 阶段有 PCR 辅助监督；简化 V2 关闭该项。
`pcr/` 是独立的影像序列分类流程，两种 V2 都可以使用它。

## 安装和运行

推荐 Python 3.11；本次基础代码与 PCR 单元检查使用 Python 3.12。
PyTorch 请按设备安装；正式生成使用 MONAI 后端。

```bash
python -m pip install -e '.[test,monai,pcr]'
python -m pytest -q
python world.py smoke --output /tmp/world-v2-smoke
python world.py --help
python pcr/run.py --help
```

真实数据训练与生成命令见 [GENERATION_GUIDE.md](GENERATION_GUIDE.md)，
现有 ROI32 的适配见 [docs/REGISTERED_ROI32_RUN.md](docs/REGISTERED_ROI32_RUN.md)。
外部 PCR 的训练、特征提取和独立预测见 [pcr/README.md](pcr/README.md)。
网络设计见 [docs/DESIGN_ZH.md](docs/DESIGN_ZH.md)，本次验证见 [docs/VALIDATION.md](docs/VALIDATION.md)。

## 可复现范围

上传包含源码、配置、测试和说明。患者数据、特征缓存、真实划分及所有训练权重均需在本地提供。
生产入口要求匹配的 VQ 和生成器权重；合成 smoke 只能验证工程链路。
历史环境路径已替换为通用占位符；历史完整队列脚本需要设置对应的数据和实验输出路径。
`GENERATION_GUIDE.md` 的原交付时期说明属于历史背景，本次整理没有重新进行完整临床训练。

生成器和外部 PCR 分类器独立训练。生成模型使用过的 102 人队列不能称为独立端到端测试集。
不要把简化 V2 保留但未训练的内部 PCR 辅助头当作外部分类器。

许可见 [LICENSE](LICENSE)、`licenses/`、`pcr/LICENSE` 和 `vendor/mewm-ispy2/LICENSE.md`。
新增入口的说明见 [docs/RELEASE.md](docs/RELEASE.md)。
