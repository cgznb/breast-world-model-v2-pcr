# 数据、时间和信息边界

## Manifest

JSON `schema="symm_world_manifest_v2"`，`phase_order=["pre_aqc0","first_post_aqc1","metadata_late"]`，`latent_channels=24`。`views` 和 `pairs` 不包含原图 bytes，只引用你本地的文件。相对路径相对于 manifest 目录。

一个 view 至少包含：

```json
{
  "id": "PATIENT_T0_SOURCE_GRID",
  "patient_id": "PATIENT",
  "visit": "T0",
  "split": "train",
  "latent": "/local/latent.npy",
  "geometry": {
    "shape_zyx": [96,256,256],
    "spacing_xyz_mm": [1.25,1.25,2.5],
    "origin_lps_mm": [0,0,0],
    "direction_lps": [1,0,0,0,1,0,0,0,1]
  },
  "grid_id": "ACTUAL_COMMON_PHYSICAL_GRID_ID",
  "source_available_grid": false
}
```

此处 origin 等仅演示格式，不可用于替代真实几何。`latent.npy` 必须为有限 float array `[24,D,H,W]`，三个相位各占连续 8 channels，且出自同一匹配的原 VQ。所有尺寸、ROI 定位、归一化、codebook 版本应由你现有数据准备流程核验。

Pair 引用 `source/target` 的 view ID，而不是任意数组路径。需有 id、patient_id、split、conditions。程序核对源/目标的 patient、visit 和 split 与 pair 的声明相符。患者不能跨 train/val/test。

`registered`、`anatomy_comparable`、`treatment_verified` 是分别的属性。不能用一个 true 代表全部。rollout 还要求 `scenario_id` 统一，且 `plan_known_at_initial_source=true`。QC sidecar 格式见 `configs/qc.example.json`。

## 临床条件

完整允许字段在 `conditioning.py` 的 `ALLOWED` 中；未列出字段报错。未来病理、pCR、生存、复发等不能混在 conditions 中。标签放在 `view.labels` 或 auxiliary sidecars，仅供训练/评价。

`stage_i` 和 `stage_j` 永远按较早→较晚声明，即使执行 reverse retrodiction，也不将药物计划简单倒序当作“解药”。未核验日期的 `delta_days` 被置为 missing，阶段 token 仍可用。

治疗分段 start/end 使用**统一患者治疗时间参考下的天数**，不能三条边分别从 0 起而又谎称可直接拼接。dose 是你事先统一的数值剂量/暴露量，不能把 mg、mg/m² 等混为一个未经单位处理的值。缺少剂量的旧数据默认不能编造 dose=1 用于研究级 treatment effect 建模。

每段必须是 `drug,start,end,dose,known_at_source` 五个字段，严格与 `configs/conditions.example.json` 以及代码检查一致。例如：

```json
{"drug":"KNOWN_REGIMEN_COMPONENT","start":0,"end":21,"dose":1.0,"known_at_source":true}
```

这个示例 dose=1 只演示格式，不是推荐剂量。相同 drug/dose 的相邻区间可归并以核对 composite/direct scenario；协议变化、未知未来加药或手术后的信息不能作为 baseline 已知条件。

## 同访视辅助监督

可选 `view.auxiliary` 指向 NPZ，使用以下键：

| 键 | 形状 | 含义 |
|---|---|---|
| kinetics | `[3,d,h,w]` | early−pre、late−pre、late−early 信号差 |
| kinetics_mask | `[1或3,d,h,w]` | 有效三相覆盖和可信监督范围，0–1 |
| segmentation | `[1,d,h,w]` | 肿瘤 soft mask，0–1；全零有意义 |
| segmentation_mask | `[1,d,h,w]` | 标注/覆盖有效范围，不是肿瘤本身 |
| external_tokens | `[N,encoder.dim]` | 指定 teacher、指定 spatial grid 的冻结特征 |

kinetics 必须同时提供 mask。默认不假设所有背景/填充都有效。只有 foreground 不等于具有可靠 tumor mask；禁止把 breast/body foreground 当作肿瘤分割金标准。

`view.labels.pcr` 为 0、1 或 null。`view.labels.biomarkers` 为四个你事先定义的量，每项允许 null；建议在训练集拟合它们的单位变换/标准化，并为每项保存定义。框架不会将这四个位置自动解释为可互换的 FTV/PE/SER 等医学量。

`cache-aux --kind measured` 从 `view.images` 的 NPZ 读取 `images=[3,D,H,W]`、`support`、可选 `tumor_mask`。support 可以为原仓库保存的 np.packbits bytes 或逐 voxel bool。Manifest 的 `image_normalization` 必须有跨三相共享的 scalar mean/std；分别归一化到不同尺度的三相不满足此契约。对 unsupported voxels 做 mask-weighted pooling，避免填充值影响信号差。

原缓存仅 val 保存 references 时，需要先为 train 导出对应的、经过 QA 的三相 image sidecars。程序只会将缺失计入 `missing`，不会伪造训练 MRI。Codec proxy 可以不要求原图，但必须写明它不是 measured。

## 文件变更与 resume

训练首次创建 metadata 时记录 config SHA-256、manifest SHA-256、训练标准化、临床词表、资产 size/mtime，辅助文件有声明 SHA 时再核对 digest。训练后改变辅助 sidecar、manifest 或 config 会拒绝 resume。先构造最终数据/teacher targets，再开始 A。

对于来源可疑或未受信任的旧 checkpoint，不要使用 `--trusted-legacy-codec`；它允许 Python pickle 的兼容加载，应仅用于你自己的已信任历史模型。

## 独立 test

数据集按患者切分是必要但非充分条件。你还需核对 frozen VQ、segmentation teacher、external semantic teacher、超参和 checkpoint selection 是否见过 test。只有确实核验之后，才在 manifest 的 provenance 声明 `end_to_end_test_isolation_verified=true`。不要因为文件夹叫 test 就认为它独立。
