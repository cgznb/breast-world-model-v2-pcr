# Code Release, 2026-09-27

Source snapshots: Symm-FM World V2; longitudinal_temporal_pillar; the MeWM-ISPY2
ROI32 dependency package. Original model/training source and all classifier
source, scripts and configurations are retained. V2 includes both the original
and simplified-A configuration. Upstream authorship and licenses are retained.

Packaging changes:

- Added a repository overview and the complete external pCR source under `pcr/`.
- Added input-only pCR bundle export, local Pillar encoding and prediction.
- Added optional PCR dependencies to the root package metadata.
- Bundled the original ROI32 dependency modules under `vendor/mewm-ispy2/`.
- Redirected the main legacy V2 PCR evaluator to these bundled source packages.
- Replaced machine-specific roots with generic placeholders; the changed file
  list is in `PATH_REDACTIONS.json`. Historical dataset references still need
  local configuration; the portable prediction CLI uses explicit input paths.
- Marked the original historical-fold integration test as requiring its private
  input artifacts when those files are absent. Its assertions are unchanged.

`GENERATION_GUIDE.md` is the original generator guide, preserved for its API and
design documentation. Reports referenced there describe the original delivery;
this release's actual checks are listed in `VALIDATION.md`.

No foundation weights, fitted cohort checkpoints, MRI/CT images, labels, patient
splits, per-patient outputs, credentials or original machine roots are published.
The new pCR adapter has synthetic parity tests; full trained-weight/real-image
end-to-end replay was not performed for this code release.
