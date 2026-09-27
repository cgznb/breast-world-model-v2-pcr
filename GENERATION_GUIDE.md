# Symm-FM World V2：三相 DCE 患者状态与纵向随机动力学

这是一个可独立执行、也可增量安装到 `cgznb/symm-fm` 的完整新版三相工作流。包含模型、数据适配、三个优化阶段、独立推理、测试、配置和研究设计说明，不是伪代码。基于已核对的上游提交 `92265d3b1749ae3b686f2089843c49da129fd4d2`。

**交付范围：完整新版三相源码，而非原仓库全部工作流的逐文件镜像。** 原仓库的 MU-Glioma、ROI32、历史实验等代码没有装入本包，也没有被删除或覆盖。本包可从现有三相 latent cache 直接训练。`scripts/install_into_repo.py` 安装增量工作流；`scripts/assemble_full_repository.py` 从你的本地原仓库生成“完整原仓库 + 新版”的 ZIP，且只使用 Git 已跟踪源码，不复制患者文件。

未包含患者 MRI、患者标签、VQ 权重或任何预训练基础模型权重。未进行真实患者训练、GPU 全尺寸训练或医学有效性验证。原 VQ 的 checkpoint 兼容性做了代码结构核对和合成 checkpoint 测试，但未读取你实际训练的 VQ 权重。测试证据见 `reports/`。

## 1. 先跑一遍完整流程

在本目录运行；已有兼容 PyTorch 环境时不要随意重装 GPU wheel。

```bash
python -m pip install -e '.[test]'
python world.py smoke --output /tmp/symm_world_v2_smoke
python -m pytest -q
```

smoke 使用真实的缩小版 3D Swin/ConvNeXt/Transformer/U-Net，而不是将生产模型替换为单层 mock。它依次训练 representation、flow、rollout，随后在合成 test split 上评价并执行不读取 target 的独立推理。合成报告不能作为医学实验结果。

生产 MONAI 后端另装：

```bash
python -m pip install 'monai==1.5.1'
```

`configs/cache_safe.yaml` 调用 MONAI 的实际 `DiffusionModelUNet`。不安装 MONAI 时可明确选择 `configs/cache_safe_native.yaml`，使用包内完整的多尺度条件 3D U-Net；程序不会静默更换后端。

## 2. 接入你现有的三相缓存

```bash
python world.py convert-legacy \
  --root /absolute/path/to/three_phase_symmflow_all_pairs_t0_v2_run \
  --output /absolute/path/to/world_v2_manifest.json

python world.py audit \
  --manifest /absolute/path/to/world_v2_manifest.json \
  --config configs/cache_safe.yaml \
  --scan-arrays \
  --output /absolute/path/to/world_v2_audit.json

python world.py train \
  --manifest /absolute/path/to/world_v2_manifest.json \
  --config configs/cache_safe.yaml \
  --stage all \
  --output /absolute/path/to/new_world_v2_run
```

此配置实际执行 A（表征）→B（Flow），不会对原生未配准缓存偷偷启用空间 rollout。完整 C 阶段使用 `configs/registered_full.yaml`，但必须先有真正通过几何、治疗计划和临床时间核验的 triplets。不能把缺失的核验布尔值改成 true 来消除报错。

原三相缓存的 `admitted_inventory.json`、`latents/` 和相应 VQ 权重是你原工作流产生的输入。没有缓存时，继续用原仓库的原生三相 prepare 生成；本包不重新实现 DICOM 下载、原始 MRI 跨时间点配准或裁剪策略。

## 3. 增加真实的 DCE/分割约束

先按 `docs/DATA_CONTRACT.md` 准备每个 visit 的三相图像和有效覆盖 mask，以及可用的肿瘤 mask。**在首次训练之前**生成辅助监督和最终 manifest：

```bash
python world.py cache-aux \
  --manifest /absolute/path/to/world_v2_manifest.json \
  --kind measured \
  --output-dir /absolute/path/to/measured_aux \
  --output-manifest /absolute/path/to/world_v2_measured_manifest.json
```

上游常见缓存只保存 validation 图像 references，未保存 training references。因此，仅运行上述命令未必得到训练患者的真实 DCE 监督；检查 audit/task_support 的数量。没有标签时对应 loss 为零，而不是将缺失当负样本。

没有原始图像但有匹配 VQ 时，`--kind codec_proxy --codec /path/to/codec.pt` 可导出明确标注的**解码重建代理**。它不等于真实 MRI 生理监督。动态图像差分 loss 的另一个选项是 `configs/decoded_proxy.yaml`，训练时也需提供匹配 `--codec`。

预训练 teacher 为可选接口，使用本地可信的 TorchScript 3D encoder：

```bash
python world.py cache-teacher \
  --manifest /absolute/path/to/world_v2_measured_manifest.json \
  --teacher /absolute/path/to/trusted_dense_teacher.ts \
  --config configs/external_teacher.yaml \
  --output-dir /absolute/path/to/teacher_aux \
  --output-manifest /absolute/path/to/world_v2_teacher_manifest.json
```

teacher 必须输出 `[B,192,D,H,W]` 的指定层特征（生产配置），由你明确导出。程序记录实际文件 SHA-256、维度和导出数量；不自动下载、不把随机初始化网络称为基础模型。本包没有实现对任意 VoCo/DINO 权重键名的自动迁移。

## 4. 分阶段、恢复和推理

```bash
python world.py train --config configs/cache_safe.yaml \
  --manifest /absolute/path/to/world_v2_manifest.json \
  --output /absolute/path/to/new_world_v2_run --stage representation

python world.py train --config configs/cache_safe.yaml \
  --manifest /absolute/path/to/world_v2_manifest.json \
  --output /absolute/path/to/new_world_v2_run --stage flow

# 恢复时配置、manifest、缓存文件必须匹配，不可偷偷改变预算或数据。
python world.py train --config configs/cache_safe.yaml \
  --manifest /absolute/path/to/world_v2_manifest.json \
  --output /absolute/path/to/new_world_v2_run --stage flow --resume
```

独立推理不接收 target 路径、真实未来病理或患者完整 manifest：

```bash
python world.py sample \
  --checkpoint /absolute/path/to/new_world_v2_run/flow/best.pt \
  --source /absolute/path/to/source_visit_latent.npy \
  --conditions /absolute/path/to/source_available_conditions.json \
  --normalization raw \
  --direction forward \
  --samples 4 --steps 20 --method heun \
  --device cuda \
  --output /absolute/path/to/prediction.npz
```

`source_visit_latent.npy` 为原 VQ 的未标准化 `[24,D,H,W]` continuous latent；程序使用 checkpoint 内的训练集统计量。输出 `latent` 为 `[K,24,D,H,W]` 未标准化 latent。加 `--codec /path/to/codec.pt` 可同时输出解码后三相 MRI。只提供真实 source；未来病理及真实目标图像不参与推理。`--direction reverse` 是 retrodiction，不是治疗逆转或组织恢复模拟。

```bash
python world.py evaluate \
  --checkpoint /absolute/path/to/new_world_v2_run/flow/best.pt \
  --manifest /absolute/path/to/world_v2_manifest.json \
  --split val --device cuda \
  --output /absolute/path/to/validation_results.json
```

原 manifest 只有 train/val 时不能凭空得到独立 test。即使另有患者级 test split，只有你核验 VQ、teacher、生成器、分类器和模型选择全链条未使用 test，并设置相应 provenance，报告才标记为端到端独立测试。

## 5. 安装进原仓库，或生成完整合并仓库 ZIP

```bash
python scripts/install_into_repo.py --repo /absolute/path/to/symm-fm
cd /absolute/path/to/symm-fm
python run_world_v2.py --help
```

新增 `workflows/world_v2/` 和 `run_world_v2.py`，原始 `run.py`、三相模型、历史配置均不覆盖。旧 checkpoint 仍使用旧入口；新版 checkpoint 使用新入口。两者不可假装是同一个网络继续 resume。

从已包含固定上游提交的本地 Git 仓库组装完整源码包：

```bash
python scripts/assemble_full_repository.py \
  --local-upstream /absolute/path/to/symm-fm \
  --output /absolute/path/to/symm-fm-world-v2-full.zip
```

也可显式用 `--clone` 代替 `--local-upstream`，在你有网络的环境中克隆公开上游后组装。该组装命令包含完整上游的既有代码及本包新增工作流，固定上游提交，不读取 Git 未跟踪的患者缓存。当前对话提供的 ZIP 本身不是这个完整上游镜像。

详细模型、数学修正、每项 loss 的条件与实现位置：`docs/DESIGN_ZH.md`。数据格式：`docs/DATA_CONTRACT.md`。实验：`docs/EXPERIMENTS_ZH.md`。公开代码来源和实际复用边界：`docs/SOURCES.md`、`SOURCE_AUDIT.json`。
