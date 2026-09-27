"""Evaluate frozen V1/V4 classifiers with World V2 images, without fitting."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
PCR = REPO.parent / "longitudinal_temporal_pillar"
ROOT = REPO / "runs/registered_roi32_20260919/analysis_30_pcr_v1v4_20260919"
BASE = PCR / "results/registered_three_phase_roi32_pcr_v1_20260917"
HEADS = PCR / "results/registered_three_phase_roi32_pcr_300epoch_patience50_20260918"
OUT = ROOT / "pcr/evaluation"
sys.path[:0] = [str(PCR), str(REPO / "src")]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
import torch

from scripts.run_full978_anti_overfit import _load_split, _atomic_csv
from scripts.run_full978_independent_cv import _canonical_split
from src.data import EmbStore
from src import registered_three_phase_optimization as shared
from src.registered_three_phase_generated_pcr import replaced_split

KEYS = ["model", "depth", "seed", "source"]
SOURCES = ("real", "copy_T0", "direct_mc4", "rollout_mc4", "previous_real_mc4")
WINDOWS = {1: "T0", 2: "T0-T1", 3: "T0-T2", 4: "T0-T3"}
LABELS = {"real": "Real MRI", "copy_T0": "Copy T0", "direct_mc4": "World direct",
          "rollout_mc4": "World rollout", "previous_real_mc4": "World previous-real"}


def read(path):
    return json.loads(path.read_text())


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, allow_nan=False) + "\n")


def identity(path):
    path = path.resolve()
    stat = path.stat()
    return {"path": str(path), "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def csv(path):
    return pd.read_csv(path, dtype={"patient_id": str}, float_precision="round_trip")


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    assert (HEADS / "TRAINING_COMPLETE.json").is_file()
    frozen = read(HEADS / "FROZEN_MODELS.json")
    shared.verify_inventory(frozen["artifacts"])
    refs = read(HEADS / "model_references.json")
    assert len(refs) == 250
    expected = {(arm, depth, seed, fold) for arm, depths in (("v1", (1, 2, 3, 4)), ("v4", (4,)))
                for depth in depths for seed in range(42, 52) for fold in range(5)}
    actual = {(r["arm"], r["depth"], r["seed"], r["outer"]) for r in refs}
    assert actual == expected
    document = {"models": [{**r, "identity": identity(Path(r["path"]) / "model.pt")} for r in refs],
                "cohort": identity(BASE / "cohort.json"), "metadata": identity(BASE / "holdout_metadata.csv"),
                "generation_protocol": identity(ROOT / "protocol.json"),
                "version": "V1/V4 matched 300 epochs maximum, patience 50; frozen validation-AUC selected heads",
                "generator": "World V2 best EMA step 4000", "classifier_optimizer_updates": 0,
                "historical_test_patients": 102, "positive": 32, "independent_end_to_end_test": False,
                "mc4": "Mean of four complete-sequence probabilities inside each classifier",
                "seed_policy": "All ten seeds 42-51 separately; no seed selection or cross-seed average",
                "threshold_policy": "0.5 fixed, no historical-label threshold tuning",
                "bootstrap": {"samples": 2000, "seed": 20260919, "method": "paired stratified patient resampling"}}
    path = OUT / "protocol.json"
    if path.exists():
        assert read(path) == document
    else:
        write(path, document)
    return refs


def raw_real():
    ids = read(BASE / "cohort.json")["split"]["val"]
    raw = _canonical_split(_load_split(BASE / "embeddings/real", BASE / "holdout_metadata.csv", ids), 4)
    assert len(raw["pids"]) == 102 and raw["labels"].sum() == 32
    return raw


def probabilities(ref, state, model, splits):
    ids = set(splits["real"]["pids"])
    task = state["task"]
    assert not ids.intersection(set(task["train_ids"]) | set(task["val_ids"]))
    assert (task["depth"], task["seed"], task["outer"]) == (ref["depth"], ref["seed"], ref["outer"])
    rows = []
    for source, split in splits.items():
        probability, _, _ = shared.predict_checkpoint(state, split, model=model)
        assert np.isfinite(probability).all() and np.all((probability >= 0) & (probability <= 1))
        rows.append(pd.DataFrame({"model": ref["arm"], "depth": ref["depth"], "seed": ref["seed"],
            "outer": ref["outer"], "source": source, "patient_id": split["pids"],
            "label": split["labels"].astype(int), "probability": probability.astype(np.float64)}))
    return pd.concat(rows, ignore_index=True)


def replay_real(frame):
    expected = csv(HEADS / "evaluation/holdout/mc4_fold_predictions.csv")
    expected = expected[expected.source == "real"]
    observed = frame[frame.source == "real"]
    keys = KEYS + ["outer", "patient_id", "label"]
    a, b = observed.set_index(keys).sort_index(), expected.set_index(keys).sort_index()
    pd.testing.assert_index_equal(a.index, b.index)
    error = float(np.max(np.abs(a.probability.to_numpy() - b.probability.to_numpy())))
    if error > 1e-7:
        raise ValueError(f"Existing real-input classifier probabilities failed replay: {error}")
    return error


def predict(refs, real_only=False):
    if not real_only and not (ROOT / "pcr/FEATURES_COMPLETE.json").is_file():
        raise ValueError("All World V2 features must complete before full evaluation")
    raw = raw_real()
    routes = read(ROOT / "protocol.json")["routes"]
    splits = {"real": raw, "copy_T0": replaced_split(raw, routes, "copy_T0")}
    if not real_only:
        for mode in ("direct", "rollout", "previous_real"):
            for draw in range(4):
                folder = ROOT / "pcr/embeddings" / mode / f"draw_{draw}"
                splits[f"{mode}_draw_{draw}"] = replaced_split(raw, routes, mode, EmbStore(str(folder)))
    parts = []
    for index, ref in enumerate(refs, 1):
        folder = OUT / ("real_cache" if real_only else "model_cache")
        name = f"{ref['arm']}_T{ref['depth'] - 1}_seed{ref['seed']}_fold{ref['outer']}.csv"
        path = folder / name
        if path.exists():
            frame = csv(path)
        else:
            state = torch.load(Path(ref["path"]) / "model.pt", map_location="cpu", weights_only=False)
            model = shared.load_model(state)
            frame = probabilities(ref, state, model, splits)
            _atomic_csv(path, frame)
        assert len(frame) == 102 * len(splits)
        parts.append(frame)
        if index % 25 == 0:
            print(json.dumps({"stage": "classifier_inference", "models": index, "total": len(refs), "real_only": real_only}), flush=True)
    frame = pd.concat(parts, ignore_index=True)
    error = replay_real(frame)
    if real_only:
        _atomic_csv(OUT / "real_and_copy_predictions.csv", frame)
        write(OUT / "real_replay.json", {"models": len(refs), "max_probability_error": error, "completed_utc": now()})
    else:
        _atomic_csv(OUT / "draw_fold_predictions.csv", frame)
    return frame, error


def metric(y, p):
    y, p = np.asarray(y).astype(int), np.asarray(p, dtype=np.float64)
    pred = p >= .5
    tp, tn = int(np.sum(pred & (y == 1))), int(np.sum(~pred & (y == 0)))
    fp, fn = int(np.sum(pred & (y == 0))), int(np.sum(~pred & (y == 1)))
    return {"auroc": float(roc_auc_score(y, p)), "ap": float(average_precision_score(y, p)),
            "brier": float(brier_score_loss(y, p)), "logloss": float(log_loss(y, p, labels=[0, 1])),
            "sensitivity": tp / (tp + fn), "specificity": tn / (tn + fp), "accuracy": (tp + tn) / len(y),
            "f1": 2 * tp / max(2 * tp + fp + fn, 1), "patients": len(y), "positive": int(y.sum())}


def mc4(raw):
    keys = [k for k in raw if k not in ("source", "probability")]
    parts = [raw[raw.source.isin(["real", "copy_T0"])].copy()]
    for mode in ("direct", "rollout", "previous_real"):
        rows = raw[raw.source.str.startswith(mode + "_draw_")]
        assert (rows.groupby(keys).size() == 4).all()
        assert set(rows.source) == {f"{mode}_draw_{i}" for i in range(4)}
        reduced = rows.groupby(keys, as_index=False).probability.mean()
        reduced["source"] = mode + "_mc4"
        parts.append(reduced)
    return pd.concat(parts, ignore_index=True)


def summarize(predictions):
    folds, summaries = [], []
    for key, group in predictions.groupby(KEYS + ["outer"]):
        assert len(group) == 102 and group.patient_id.is_unique and group.label.sum() == 32
        folds.append({**dict(zip(KEYS + ["outer"], key)), **metric(group.label, group.probability)})
    folds = pd.DataFrame(folds)
    ensemble = predictions.groupby(KEYS + ["patient_id", "label"], as_index=False).probability.mean()
    ensemble_scores = pd.DataFrame([{**dict(zip(KEYS, key)), **metric(g.label, g.probability)}
                                   for key, g in ensemble.groupby(KEYS)])
    for key, group in folds.groupby(KEYS):
        assert set(group.outer) == set(range(5)) and len(group) == 5
        row = {**dict(zip(KEYS, key)), "window": WINDOWS[key[1]], "patients": 102, "positive": 32}
        for field in ("auroc", "ap", "brier", "logloss", "sensitivity", "specificity", "accuracy", "f1"):
            row[field + "_mean"] = float(group[field].mean())
            row[field + "_fold_sd"] = float(group[field].std(ddof=1))
        row.update({f"F{r.outer + 1}_auroc": float(r.auroc) for r in group.itertuples()})
        summaries.append(row)
    summary = pd.DataFrame(summaries).merge(ensemble_scores[KEYS + ["auroc"]].rename(
        columns={"auroc": "probability_ensemble_auroc"}), on=KEYS, validate="one_to_one")
    for name, frame in (("mc4_fold_predictions", predictions), ("fold_metrics", folds), ("seed_summary", summary),
                        ("probability_ensemble_predictions", ensemble), ("probability_ensemble_metrics", ensemble_scores)):
        _atomic_csv(OUT / (name + ".csv"), frame)
    return folds, summary, ensemble, ensemble_scores


def rank_auc(labels, probabilities):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    positive, negative = int(labels.sum()), int((1 - labels).sum())
    return float((rankdata(probabilities)[labels == 1].sum() - positive * (positive + 1) / 2) / (positive * negative))


def verify(raw, reduced, folds, summary, ensemble):
    errors = []
    expected_groups = {(model, depth, seed, source, fold) for model, depths in (("v1", range(1, 5)), ("v4", (4,)))
                       for depth in depths for seed in range(42, 52) for source in SOURCES for fold in range(5)}
    assert set(folds[KEYS + ["outer"]].itertuples(index=False, name=None)) == expected_groups
    assert len(raw) == 357000 and len(reduced) == 127500
    reduced_lookup = reduced.set_index(KEYS + ["outer", "patient_id", "label"]).probability
    keys = ["model", "depth", "seed", "outer", "patient_id", "label"]
    for mode in ("direct", "rollout", "previous_real"):
        matrix = raw[raw.source.str.startswith(mode + "_draw_")].pivot(index=keys, columns="source", values="probability")
        assert matrix.shape == (25500, 4) and not matrix.isna().any().any()
        actual = reduced[reduced.source == mode + "_mc4"].set_index(keys).loc[matrix.index, "probability"]
        errors.append(float(np.max(np.abs(matrix.to_numpy().sum(axis=1) / 4 - actual.to_numpy()))))
    score_lookup = folds.set_index(KEYS + ["outer"])
    for key, part in reduced.groupby(KEYS + ["outer"]):
        errors.append(abs(rank_auc(part.label, part.probability) - score_lookup.loc[key, "auroc"]))
    summary_lookup = summary.set_index(KEYS)
    for key, part in reduced.groupby(KEYS):
        matrix = part.pivot(index="patient_id", columns="outer", values="probability")
        assert matrix.shape == (102, 5)
        actual = ensemble.set_index(KEYS).loc[key].set_index("patient_id").loc[matrix.index]
        probability = matrix.to_numpy().sum(axis=1) / 5
        errors.append(float(np.max(np.abs(probability - actual.probability.to_numpy()))))
        errors.append(abs(rank_auc(actual.label, probability) - summary_lookup.loc[key, "probability_ensemble_auroc"]))
        scores = score_lookup.loc[key].auroc.to_numpy()
        errors.extend([abs(scores.mean() - summary_lookup.loc[key, "auroc_mean"]),
                       abs(scores.std(ddof=1) - summary_lookup.loc[key, "auroc_fold_sd"])])
    for depth, sources in ((1, SOURCES), (2, SOURCES[2:])):
        part = reduced[(reduced.depth == depth) & reduced.source.isin(sources)]
        matrix = part.pivot(index=keys, columns="source", values="probability")
        errors.append(float(np.max(np.ptp(matrix.to_numpy(), axis=1))))
    assert max(errors) < 1e-12
    return {"passed": True, "independent_auc_replays": len(folds), "ensemble_auc_replays": len(summary),
            "raw_rows": len(raw), "mc4_rows": len(reduced), "max_metric_or_probability_error": max(errors),
            "identical_T0_and_T1_controls": True}


def paired_intervals(predictions):
    part = predictions[predictions.depth == 4]
    ids = sorted(part.patient_id.unique())
    labels = part.drop_duplicates("patient_id").set_index("patient_id").loc[ids, "label"].to_numpy()
    npos, nneg = int(labels.sum()), int((1 - labels).sum())
    rng = np.random.default_rng(20260919)
    wp = rng.multinomial(npos, np.full(npos, 1 / npos), 2000)
    wn = rng.multinomial(nneg, np.full(nneg, 1 / nneg), 2000)
    rows = []
    for seed, group in part.groupby("seed"):
        matrices = {(model, source): g.pivot(index="outer", columns="patient_id", values="probability").loc[range(5), ids].to_numpy()
                    for (model, source), g in group.groupby(["model", "source"])}
        for statistic in ("fold_mean", "probability_ensemble"):
            kernels = {}
            for key, matrix in matrices.items():
                p = matrix if statistic == "fold_mean" else matrix.mean(0, keepdims=True)
                delta = p[:, labels == 1, None] - p[:, None, labels == 0]
                kernels[key] = ((delta > 0).astype(float) + .5 * (delta == 0)).mean(0)
            comparisons = [(f"{model}:{source}-real", (model, source), (model, "real"))
                           for model in ("v1", "v4") for source in SOURCES[2:]]
            comparisons += [(f"{model}:{source}-copy_T0", (model, source), (model, "copy_T0"))
                            for model in ("v1", "v4") for source in SOURCES[2:]]
            comparisons += [(f"v4-v1:{source}", ("v4", source), ("v1", source)) for source in SOURCES]
            for label, a, b in comparisons:
                difference = kernels[a] - kernels[b]
                samples = ((wp @ difference) * wn).sum(axis=1) / (npos * nneg)
                lo, hi = np.quantile(samples, [.025, .975])
                rows.append({"seed": seed, "statistic": statistic, "comparison": label,
                             "auc_difference": float(difference.mean()), "ci95_low": float(lo), "ci95_high": float(hi)})
    result = pd.DataFrame(rows)
    _atomic_csv(OUT / "paired_t3_intervals.csv", result)
    return result


def old_comparison(summary):
    old = csv(HEADS / "evaluation/holdout/seed_summary.csv")
    columns = KEYS + ["auroc_mean", "probability_ensemble_auroc"]
    joined = summary[columns].merge(old[columns], on=KEYS, how="inner", suffixes=("_world_v2", "_old_generator"), validate="one_to_one")
    for field in ("auroc_mean", "probability_ensemble_auroc"):
        joined[field + "_difference"] = joined[field + "_world_v2"] - joined[field + "_old_generator"]
    _atomic_csv(OUT / "world_v2_vs_old_generator.csv", joined)
    return joined


def plot_results(summary):
    colors = {"real": "#257657", "copy_T0": "#888888", "direct_mc4": "#bc5368",
              "rollout_mc4": "#4d76ad", "previous_real_mc4": "#b4942c"}
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained", sharex=True, sharey=True)
    for row, model in enumerate(("v1", "v4")):
        for col, (field, label) in enumerate((("auroc_mean", "Mean of five model AUCs"),
                                             ("probability_ensemble_auroc", "AUC of five-model mean probability"))):
            ax = axes[row, col]
            for source in SOURCES:
                data = summary[(summary.model == model) & (summary.depth == 4) & (summary.source == source)].sort_values("seed")
                ax.plot(data.seed, data[field], marker="o", markersize=4, linewidth=1.4, color=colors[source], label=LABELS[source])
            ax.set_title(model.upper() + " T0-T3 | " + label, fontsize=11)
            ax.set_xticks(range(42, 52))
            ax.set_ylabel("Historical test AUROC")
            ax.set_xlabel("Classifier seed")
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5)
    fig.suptitle("Frozen V1/V4 (300/50) | 102 patients, 32 positive | World V2 EMA 4000", fontsize=13)
    fig.savefig(OUT / "t3_auc_by_seed.png", dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 4, figsize=(17, 4.6), layout="constrained", sharey=True)
    for depth, ax in enumerate(axes, 1):
        for source in SOURCES:
            data = summary[(summary.model == "v1") & (summary.depth == depth) & (summary.source == source)].sort_values("seed")
            ax.plot(data.seed, data.auroc_mean, marker="o", markersize=3, color=colors[source], linewidth=1.1, label=LABELS[source])
        ax.set_title("V1 " + WINDOWS[depth], fontsize=11)
        ax.set_xlabel("Classifier seed")
        ax.set_xticks([42, 45, 48, 51])
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Mean of five model AUCs")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=5)
    fig.suptitle("V1 temporal windows | same 102 historical patients, retained visit masks", fontsize=13)
    fig.savefig(OUT / "v1_windows.png", dpi=160)
    plt.close(fig)


def table(frame, field, sd=False):
    matrix = frame.pivot(index="seed", columns="source", values=field).loc[range(42, 52), list(SOURCES)]
    lines = ["| Seed | Real | Copy T0 | World direct | World rollout | World previous-real |",
             "|---:|---:|---:|---:|---:|---:|"]
    for seed, row in matrix.iterrows():
        cells = []
        for source in SOURCES:
            cell = f"{row[source]:.4f}"
            if sd:
                deviation = frame[(frame.seed == seed) & (frame.source == source)].auroc_fold_sd.iloc[0]
                cell += f" +/- {deviation:.4f}"
            cells.append(cell)
        lines.append(f"| {seed} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def export(summary, folds, ensemble_scores, intervals, comparisons, verification):
    lines = ["# World V2 MRI to frozen V1/V4 pCR evaluation", "",
             "[Interactive results](index.html) | [Chinese interpretation](../../results_zh.md)", "",
             "This evaluates the latest matched V1/V4 classifiers (maximum 300 epochs, patience 50). "
             "All 250 existing classifiers are frozen: V1 has four independently fitted temporal windows, "
             "and V4 has only T0-T3. No pCR classifier was refitted; no threshold or seed was chosen on these results.", "",
             "The 102 historical test patients (32 pCR-positive) were excluded from classifier fitting. "
             "They were also used for World V2 validation and checkpoint selection, so this is not an "
             "independent end-to-end test. The recorded follow-up stages and intervals are retrospective covariates. "
             "Each of the five fold-trained models evaluates the same 102 patients; these are not five disjoint test subsets.", "",
             "[Workbook](pcr_world_v2_v1_v4.xlsx) | [All seeds and folds](seed_summary.csv) | "
             "[Per-model metrics](fold_metrics.csv) | [Ensemble metrics](probability_ensemble_metrics.csv) | "
             "[Paired intervals](paired_t3_intervals.csv) | [Old generator comparison](world_v2_vs_old_generator.csv)", "",
             "![T3 seed results](t3_auc_by_seed.png)", "",
             "## Input policies", "",
             "- Real: all available real MRI visits in the model window.",
             "- Copy T0: repeat the real T0 Pillar embedding in every retained future slot.",
             "- World direct: real T0 independently forecasts each retained future visit.",
             "- World rollout: real T0 forecasts T1, generated T1 forecasts T2, then generated T2 forecasts T3; "
             "draw identities remain paired along each trajectory. World Stage C was disabled, so consistency across rollout steps was not explicitly trained.",
             "- World previous-real: the real preceding visit forecasts the next visit. This uses more observed information than T0-only direct/rollout.", "",
             "Every generated MRI has three phases at 32 x 128 x 128. World uses Heun25 and four draws. "
             "Frozen FP32 Pillar produces a 1152-dimensional feature per visit. Four complete-sequence "
             "pCR probabilities are averaged inside each classifier (MC4); MRI or embedding means are not classifier inputs. "
             "The five resulting classifier probabilities are optionally averaged to form the probability ensemble. "
             "The mean of five individual AUCs and the AUC of mean probability are different statistics.", "",
             "V4 differs from matched V1 at T0-T3 by a 0.1 penalty on the squared residual logit during training. "
             "At T0-T3 both use an unconstrained learned scalar multiplying the raw image residual, added to the training-fold clinical prior. "
             "Only the V1 T0/T0-T1 windows use the bounded tanh residual. All ten seeds are shown separately.", ""]
    for model, depths in (("v1", range(1, 5)), ("v4", (4,))):
        for depth in depths:
            part = summary[(summary.model == model) & (summary.depth == depth)]
            lines += [f"## {model.upper()} {WINDOWS[depth]}", "", "Five-model AUC mean +/- sample SD:", "",
                      table(part, "auroc_mean", sd=True), "", "Five-model probability ensemble AUC:", "",
                      table(part, "probability_ensemble_auroc"), ""]
    lines += ["## Matched comparisons", "",
              "Paired 95% intervals use 2000 stratified patient resamples within each seed. They are exploratory, "
              "do not adjust for the many comparisons, and do not represent uncertainty across training cohorts. "
              "Sensitivity, specificity, F1, accuracy, AP, Brier score and log loss are exported. Thresholded metrics use 0.5.", "",
              "The old-generator comparison uses the same 300/50 classifier checkpoints and historical patients. "
              "It is a generator comparison, not a comparison between separately trained pCR heads. "
              "No improvement establishes independent clinical validity.", "", "## Verification", "",
              f"Real-input prediction replay maximum error: {verification['real_prediction_replay_max_error']:.3g}. "
              f"Independent rank-based verification replayed {verification['independent_auc_replays']} fold AUCs and "
              f"{verification['ensemble_auc_replays']} ensemble AUCs. "
              f"Maximum metric/aggregation error: {verification['max_metric_or_probability_error']:.3g}. "
              "T0 probabilities are identical across input policies; shared T1 generated inputs agree. "
              "Frozen model and original World runtime identities were checked.", ""]
    (OUT / "report.md").write_text("\n".join(lines))
    sheets = {"All_seed_summary": summary, "All_fold_metrics": folds, "Ensemble_metrics": ensemble_scores,
              "Paired_T3_intervals": intervals, "Old_generator_comparison": comparisons}
    for model, depths in (("v1", range(1, 5)), ("v4", (4,))):
        for depth in depths:
            part = summary[(summary.model == model) & (summary.depth == depth)]
            for field, name in (("auroc_mean", "foldmean"), ("probability_ensemble_auroc", "ensemble")):
                sheets[f"{model.upper()}_{WINDOWS[depth]}_{name}"] = part.pivot(index="seed", columns="source", values=field).reset_index()
    with pd.ExcelWriter(OUT / "pcr_world_v2_v1_v4.xlsx", engine="openpyxl") as writer:
        for name, frame in sheets.items():
            frame.to_excel(writer, sheet_name=name, index=False)
            sheet = writer.sheets[name]
            sheet.freeze_panes = "A2"
            sheet.auto_filter.ref = sheet.dimensions
            for column in sheet.columns:
                sheet.column_dimensions[column[0].column_letter].width = min(42, max(13, len(str(column[0].value)) + 2))
    workbook = pd.ExcelFile(OUT / "pcr_world_v2_v1_v4.xlsx")
    assert set(workbook.sheet_names) == set(sheets)
    for name, expected in sheets.items():
        actual = pd.read_excel(workbook, sheet_name=name)
        pd.testing.assert_frame_equal(actual, expected.reset_index(drop=True), check_dtype=False, check_names=False, atol=1e-12, rtol=1e-12)
    payload = {"summary": summary.to_dict("records"), "folds": folds.to_dict("records"),
               "ensemble": ensemble_scores.to_dict("records")}
    (OUT / "results.js").write_text("window.PCR_RESULTS=" + json.dumps(payload, ensure_ascii=True, allow_nan=False) + ";\n")
    shutil.copyfile(REPO / "scripts/world_v2_review_web/pcr.html", OUT / "index.html")
    write(OUT / "verification.json", {**verification, "workbook_sheets_replayed": len(sheets)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-only", action="store_true")
    parser.add_argument("--detach", action="store_true")
    args = parser.parse_args()
    if args.detach:
        OUT.mkdir(parents=True, exist_ok=True)
        command = [sys.executable, "-u", "-B", str(Path(__file__).resolve())]
        if args.real_only:
            command.append("--real-only")
        with (OUT / "evaluation.log").open("a") as log:
            process = subprocess.Popen(command, cwd=REPO, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True, env=os.environ.copy())
        print(json.dumps({"pid": process.pid, "log": str(OUT / "evaluation.log")}), flush=True)
        return
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mha.set_fastpath_enabled(False)
    refs = freeze()
    raw, replay_error = predict(refs, args.real_only)
    if args.real_only:
        print(json.dumps({"status": "real_replay_complete", "max_error": replay_error}), flush=True)
        return
    reduced = mc4(raw)
    folds, summary, ensemble, ensemble_scores = summarize(reduced)
    verification = verify(raw, reduced, folds, summary, ensemble)
    verification["real_prediction_replay_max_error"] = replay_error
    intervals = paired_intervals(reduced)
    comparisons = old_comparison(summary)
    plot_results(summary)
    freeze()
    export(summary, folds, ensemble_scores, intervals, comparisons, verification)
    write(OUT / "COMPLETE.json", {"completed_utc": now(), "models": len(refs), "patients": 102,
          "classifier_optimizer_updates": 0, "verification": identity(OUT / "verification.json"),
          "summary": identity(OUT / "seed_summary.csv"), "raw_predictions": identity(OUT / "draw_fold_predictions.csv")})
    print(json.dumps({"status": "complete", "models": len(refs), "verification": verification}), flush=True)


if __name__ == "__main__":
    main()
