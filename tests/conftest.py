from pathlib import Path
import pytest
import torch
from symm_world.config import load_config
from symm_world.synthetic import create_synthetic
from symm_world.data import DatasetStore
from symm_world.conditioning import ConditionSchema
from symm_world.models import RepresentationSystem

ROOT=Path(__file__).resolve().parents[1]

@pytest.fixture(autouse=True)
def threads():
    torch.set_num_threads(2)

@pytest.fixture
def cfg():
    return load_config(ROOT/"configs/smoke.yaml")

@pytest.fixture
def store(tmp_path):
    s=DatasetStore(create_synthetic(tmp_path/"data"));s.fit_statistics();return s

@pytest.fixture
def rep(cfg,store):
    return RepresentationSystem(cfg,ConditionSchema.fit(store.records("train")),store.statistics)
