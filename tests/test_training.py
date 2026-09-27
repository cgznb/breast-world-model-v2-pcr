import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from symm_world.training import train_stage,load_inference
from symm_world.utils import load_checkpoint
from symm_world.cli import sample_checkpoint
from symm_world.evaluation import evaluate_model


def assert_tree_equal(a,b):
    if isinstance(a,torch.Tensor):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_tree_equal(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for x,y in zip(a,b):assert_tree_equal(x,y)
    else:assert a==b


def test_three_stage_actual_native_training_and_source_only_sample(cfg,store,tmp_path):
    out=tmp_path/'run'
    for stage in ('representation','flow','rollout'):
        result=train_stage(stage,cfg,store,out)
        assert result['complete']
        metrics=[json.loads(x) for x in (out/stage/'metrics.jsonl').read_text().splitlines()]
        assert all(r['gradient_norm']>0 for r in metrics)
    checkpoint=out/'rollout/last.pt'
    model,meta=load_inference(checkpoint)
    assert all(not p.requires_grad for p in model.parameters())
    store.set_statistics(meta['statistics'])
    evaluation=evaluate_model(model,store,'test',meta)
    assert evaluation['independent_test'] and evaluation['samples']==2
    pair=store.records('test')[0]
    # Delete true future: forecast MUST still work.
    store.resolve(store.views[pair['target']]['latent']).unlink()
    pred=sample_checkpoint(checkpoint,store.resolve(store.views[pair['source']]['latent']),pair['conditions'],tmp_path/'prediction.npz')
    assert pred['source_only'] and pred['target_reads']==0
    with np.load(tmp_path/'prediction.npz') as arr:
        assert arr['latent'].shape==(2,24,4,8,8)
        assert np.isfinite(arr['latent']).all()
    backward=sample_checkpoint(checkpoint,store.resolve(store.views[pair['source']]['latent']),pair['conditions'],tmp_path/'reverse.npz',direction=-1)
    assert backward['direction']=='retrodiction' and 'source_landmark_pcr_probability' not in backward


def test_exact_representation_resume(cfg,store,tmp_path):
    a=tmp_path/'continuous';b=tmp_path/'resumed'
    train_stage('representation',cfg,store,a)
    partial=train_stage('representation',cfg,store,b,stop_after=1);assert not partial['complete']
    train_stage('representation',cfg,store,b,resume=True)
    x=load_checkpoint(a/'representation/last.pt');y=load_checkpoint(b/'representation/last.pt')
    for k in ('model','optimizer','scheduler','rng','step','best'):assert_tree_equal(x[k],y[k])


def test_exact_flow_resume(cfg,store,tmp_path):
    import shutil
    a=tmp_path/'continuous';b=tmp_path/'resumed'
    train_stage('representation',cfg,store,a)
    shutil.copytree(a,b)
    train_stage('flow',cfg,store,a)
    train_stage('flow',cfg,store,b,stop_after=1)
    train_stage('flow',cfg,store,b,resume=True)
    x=load_checkpoint(a/'flow/last.pt');y=load_checkpoint(b/'flow/last.pt')
    for k in ('model','teacher','optimizer','scheduler','rng','step','best'):assert_tree_equal(x[k],y[k])


def test_no_random_pcr_claim_without_labels(cfg,store,tmp_path):
    from symm_world.utils import write_json
    from symm_world.data import DatasetStore
    d=copy.deepcopy(store.manifest)
    for v in d['views']:
        v.pop('labels',None)
        for k in ('latent','auxiliary'):v[k]=str(store.resolve(v[k]))
    path=tmp_path/'no_labels.json';write_json(path,d);s=DatasetStore(path)
    root=tmp_path/'unlabeled'
    train_stage('representation',cfg,s,root)
    train_stage('flow',cfg,s,root)
    _,meta=load_inference(root/'flow/last.pt')
    assert not meta['trained_tasks']['pcr']


def test_no_silent_training_with_invalid_rollout(cfg,store,tmp_path):
    cfg.rollout.enabled=False
    with pytest.raises(ValueError):train_stage('rollout',cfg,store,tmp_path/'run')


def test_exact_rollout_resume(cfg,store,tmp_path):
    import shutil
    a=tmp_path/'continuous';b=tmp_path/'resumed'
    train_stage('representation',cfg,store,a);train_stage('flow',cfg,store,a)
    shutil.copytree(a,b)
    train_stage('rollout',cfg,store,a)
    train_stage('rollout',cfg,store,b,stop_after=1)
    train_stage('rollout',cfg,store,b,resume=True)
    x=load_checkpoint(a/'rollout/last.pt');y=load_checkpoint(b/'rollout/last.pt')
    for k in ('model','teacher','optimizer','scheduler','rng','step','best'):assert_tree_equal(x[k],y[k])
