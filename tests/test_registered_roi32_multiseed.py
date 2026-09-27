import importlib.util
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest
import torch

from symm_world.config import load_config
from symm_world.training import stage_budget
from symm_world.utils import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("roi32_multiseed", ROOT / "scripts/run_registered_roi32_multiseed.py")
queue = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queue)
CONFIG = ROOT / "configs/ablations/registered_roi32_5090_simple_a.yaml"


@pytest.fixture
def study(tmp_path, monkeypatch):
    monkeypatch.setattr(queue.shutil, "disk_usage", lambda _path: SimpleNamespace(free=64 * 1024**3))
    manifest = tmp_path / "manifest.json"
    write_json(manifest, {"synthetic_fixture": True})
    root = tmp_path / "study"
    protocol = queue.prepare(CONFIG, manifest, root, [42, 43])
    return root, protocol


def finish(folder, cfg):
    write_json(folder / "progress.json", {"status": "complete"})
    for stage in queue.STAGES:
        budget = stage_budget(cfg, stage)
        write_json(folder / stage / "status.json", {
            "complete": True, "optimizer_steps": budget["steps"],
            "sampled_pairs": budget["samples"], "trained_tasks": {"pcr": False}})
        value = {"stage": stage, "step": budget["steps"], "best": .5,
                 "metadata": {"config": cfg.to_dict(), "trained_tasks": {"pcr": False}},
                 "model": {"fixture": torch.ones(1)}}
        for kind in ("best", "last"):
            torch.save(value, folder / stage / f"{kind}.pt")


class FinishedProcess:
    pid = 999999

    def __init__(self, code):
        self.code = code

    def wait(self, timeout=None):
        return self.code

    def poll(self):
        return self.code


def test_seed_configs_are_independent_and_five_loss(study):
    root, protocol = study
    configs = [load_config(x["path"]) for x in protocol["seed_configs"]]
    assert [c.training.seed for c in configs] == [42, 43]
    assert all(c.loss.pcr == 0 and not c.rollout.enabled for c in configs)
    assert all(stage_budget(c, "representation")["steps"] == 625 for c in configs)
    assert all(stage_budget(c, "flow")["steps"] == 6000 for c in configs)
    identities = list(protocol["seed_configs"])
    assert queue.prepare(CONFIG, protocol["manifest"]["path"], root, [42, 43])["seed_configs"] == identities
    queue.verify_bindings(protocol)


def test_duplicate_seeds_and_config_changes_rejected(study):
    root, protocol = study
    with pytest.raises(ValueError, match="distinct"):
        queue.prepare(CONFIG, protocol["manifest"]["path"], root, [42, 42])
    path = Path(protocol["seed_configs"][0]["path"])
    value = read_json(path)
    value["loss"]["pcr"] = .05
    write_json(path, value)
    with pytest.raises(ValueError, match="changed"):
        queue.prepare(CONFIG, protocol["manifest"]["path"], root, [42, 43])
    with pytest.raises(ValueError, match="changed"):
        queue.verify_bindings(protocol)


def test_queue_runs_both_stages_per_seed_and_completed_resume_is_noop(study, monkeypatch):
    root, protocol = study
    calls = []

    def start(command, **kwargs):
        cfg = load_config(command[command.index("--config") + 1])
        folder = Path(command[command.index("--output") + 1])
        assert command[command.index("--stage") + 1] == "all"
        if calls:
            assert queue.completion(root / "seed_42", load_config(protocol["seed_configs"][0]["path"]))
        calls.append(cfg.training.seed)
        finish(folder, cfg)
        return FinishedProcess(0)

    monkeypatch.setattr(queue.subprocess, "Popen", start)
    result = queue.run_queue(root, protocol)
    assert calls == [42, 43]
    assert result["status"] == "complete" and result["completed_seeds"] == 2
    assert (root / "COMPLETE.json").exists()
    with pytest.raises(FileExistsError, match="resume"):
        queue.run_queue(root, protocol)
    calls.clear()
    assert queue.run_queue(root, protocol, resume=True)["completed_seeds"] == 2
    assert not calls


def test_failed_seed_stops_queue_and_resume_skips_completed_seed(study, monkeypatch):
    root, protocol = study
    finish(root / "seed_42", load_config(protocol["seed_configs"][0]["path"]))
    calls = []

    def fail(command, **kwargs):
        calls.append(command)
        return FinishedProcess(1)

    monkeypatch.setattr(queue.subprocess, "Popen", fail)
    with pytest.raises(RuntimeError, match="Seed 43 did not complete"):
        queue.run_queue(root, protocol, resume=True)
    progress = read_json(root / "progress.json")
    assert progress["status"] == "failed" and progress["completed_seeds"] == 1
    assert len(calls) == 1 and "--resume" in calls[0]
    assert not (root / "COMPLETE.json").exists()


def test_missing_or_wrong_seed_checkpoint_cannot_be_complete(study):
    root, protocol = study
    cfg = load_config(protocol["seed_configs"][0]["path"])
    folder = root / "seed_42"
    finish(folder, cfg)
    path = folder / "flow/best.pt"
    value = torch.load(path, weights_only=True)
    value["metadata"]["config"]["training"]["seed"] = 43
    torch.save(value, path)
    with pytest.raises(ValueError, match="Checkpoint"):
        queue.completion(folder, cfg)


def test_duplicate_queue_lock_is_rejected(tmp_path):
    with queue.acquire_lock(tmp_path):
        with pytest.raises(RuntimeError, match="running queue"):
            queue.acquire_lock(tmp_path)
    with queue.acquire_lock(tmp_path):
        pass


def test_low_disk_prevents_starting_a_seed(study, monkeypatch):
    root, protocol = study
    monkeypatch.setattr(queue.shutil, "disk_usage", lambda _path: SimpleNamespace(free=0))
    monkeypatch.setattr(queue.subprocess, "Popen", lambda *a, **k: pytest.fail("Started with no free space"))
    with pytest.raises(RuntimeError, match="15 GiB"):
        queue.run_queue(root, protocol)
    assert read_json(root / "progress.json")["completed_seeds"] == 0


def test_stop_during_child_creation_is_forwarded_and_does_not_advance(study, monkeypatch):
    root, protocol = study
    signals = []
    calls = []

    class StartingProcess:
        pid = 999999

        def poll(self):
            return 0 if signals else None

        def terminate(self):
            signals.append(signal.SIGTERM)

        def wait(self, timeout=None):
            assert signals
            return 0

    def start(command, **kwargs):
        calls.append(command)
        signal.getsignal(signal.SIGTERM)(signal.SIGTERM, None)
        return StartingProcess()

    monkeypatch.setattr(queue.subprocess, "Popen", start)
    result = queue.run_queue(root, protocol)
    assert result["status"] == "stopped" and len(calls) == 1
    assert result["seeds"][1]["status"] == "pending"
    assert signals == [signal.SIGTERM]
