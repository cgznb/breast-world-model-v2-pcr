# Release Validation

Date: 2026-09-27. Local Python 3.12.13, PyTorch 2.12.1+cu130. Tests and generator
smoke ran on CPU with CUDA hidden; no formal study was retrained.

| Check | Result |
| --- | --- |
| Generator `python -m pytest -q` | 75 passed |
| `world.py smoke` | Exit 0; synthetic representation, flow, rollout and source-only generation |
| PCR budget, ROI32 input, generated-PCR and release adapter tests | 20 passed, 1 skipped |
| Input-only pCR export/prediction parity | Included above; original TDN probability replay, draw/fold averaging, temporal masking and mixed-seed rejection |
| `pcr/run.py --help` | Exit 0 |

The skipped PCR test requires original patient folds and baseline recipes that
are deliberately not included in a code-only repository. This is not a passed
real-cohort integration check. The remaining historical PCR tests were not run
as a full suite because some depend on earlier private experiment artifacts.

PyTorch emitted existing JIT deprecation and prototype nested-tensor warnings.
They were not suppressed. Actual Pillar weight loading, full-size patient MRI
generation, CUDA training and trained clinical prediction are not validated by
these synthetic checks.
