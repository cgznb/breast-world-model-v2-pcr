# Other references

ConvNeXt block design: Copyright (c) Meta Platforms, Inc. and affiliates. Reference implementation https://github.com/facebookresearch/ConvNeXt/blob/main/models/convnext.py . The release uses a 3D adaptation of the mathematical block and credits the original design. No pretrained parameters are included.

I-JEPA, V-JEPA 2, VICReg, REPA and Consistency Flow Matching are credited for the specific algorithmic ideas described in `docs/SOURCES.md`. Their repositories and checkpoints are not republished as if they were part of this package. Local predictors, objective composition, data contracts and training loops are new adaptations rather than claims of exact reproductions.

The optional MONAI backend imports the installed MONAI package; use of that package and its distribution is subject to MONAI's own license. This release neither bundles MONAI nor overrides its terms.
