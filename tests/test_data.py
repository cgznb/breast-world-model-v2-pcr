import copy
import json
import numpy as np
import pytest
import torch
from symm_world.data import DatasetStore,PatientBalancedSampler
from symm_world.conditioning import validate_condition,ConditionSchema
from symm_world.losses import representation_loss,auxiliary_losses
from symm_world.evaluation import summarize_rows
from symm_world.config import from_dict
from symm_world.utils import write_json


def changed_store(store,tmp_path,fn):
    d=copy.deepcopy(store.manifest)
    for v in d['views']:
        for k in ('latent','auxiliary'):
            if v.get(k):v[k]=str(store.resolve(v[k]))
    fn(d);path=tmp_path/'mutated.json';write_json(path,d);return DatasetStore(path)

@pytest.mark.parametrize('name',['pcr','pathology','survival','target_image','future_chemotherapy_actual'])
def test_future_information_rejected(name):
    with pytest.raises(ValueError):validate_condition({'stage_i':'T0','stage_j':'T1',name:1})

def test_unverified_time_is_not_clinical_time():
    c=validate_condition({'delta_days':1234,'interval_verified':False})
    assert c['delta_days'] is None
    with pytest.raises(ValueError):validate_condition({'delta_days':0,'interval_verified':True})

def test_future_actual_treatment_rejected():
    event={'drug':'a','start':0,'end':30,'dose':1,'known_at_source':False}
    with pytest.raises(ValueError):validate_condition({'action_segments':[event]})

def test_train_only_vocabulary(store):
    with pytest.raises(ValueError):ConditionSchema.fit(store.records('val'))
    schema=ConditionSchema.fit(store.records('train'))
    assert schema.fit_split=='train'

def test_validation_data_cannot_change_training_normalization(store):
    before=copy.deepcopy(store.fit_statistics())
    for v in store.views.values():
        if v['split']!='train':
            path=store.resolve(v['latent']);x=np.load(path);np.save(path,x*100+1000)
    assert before==store.fit_statistics()

def test_patient_split_overlap_rejected(store,tmp_path):
    def mutate(d):d['views'][0]['split']='test'
    with pytest.raises(ValueError):changed_store(store,tmp_path,mutate)

def test_verified_triplets_exist(store,cfg):
    triplets,reject=store.triplets(cfg.rollout)
    assert len(triplets)==4 and not reject

@pytest.mark.parametrize('reason,mutate',[
 ('incompatible_coordinate_grids',lambda d:d['views'][0].update(grid_id='bad')),
 ('incompatible_coordinate_grids',lambda d:d['views'][0].update(geometry={'tampered':True})),
 ('registration_unverified',lambda d:d['pairs'][0].update(registered=False)),
 ('future_dependent_or_unverified_output_grid',lambda d:d['views'][0].update(source_available_grid=False)),
 ('nonadditive_clinical_intervals',lambda d:d['pairs'][0]['conditions'].update(delta_days=12)),
 ('scenario_not_declared_at_initial_source',lambda d:d['pairs'][2].update(plan_known_at_initial_source=False)),
 ('incompatible_treatment_schedules',lambda d:d['pairs'][0]['conditions']['action_segments'][0].update(drug='unplanned')),
 ('inconsistent_baseline_information',lambda d:d['pairs'][0]['conditions'].update(age=999))])
def test_triplet_gates(store,tmp_path,cfg,reason,mutate):
    s=changed_store(store,tmp_path,mutate);triples,rejected=s.triplets(cfg.rollout)
    assert len(triples)==3 and rejected.get(reason)==1

def test_patient_sampler_resume(store):
    records=store.records('train')
    a=PatientBalancedSampler(records,123);b=PatientBalancedSampler(records,123)
    assert a.batch(204,5)==b.batch(204,5)

def test_missing_labels_are_not_negatives(rep,store,cfg):
    batch=store.pair_batch(store.records('train')[:1],cfg.encoder.token_grid)
    batch['source_aux']=[{}];batch['target_aux']=[{}]
    loss,metrics,_=representation_loss(rep,batch)
    assert metrics['pcr']==0 and metrics['label_count/pcr']==0
    assert metrics['kinetics']==0 and torch.isfinite(loss)

def test_no_tumor_is_valid_not_missing(rep,store,cfg):
    x=store.pair_batch(store.records('train')[:1],cfg.encoder.token_grid)['source']
    state=rep.encoder(x)
    aux={'segmentation':torch.zeros(1,*cfg.encoder.token_grid),'segmentation_mask':torch.ones(1,*cfg.encoder.token_grid)}
    loss,count=auxiliary_losses(rep.heads,state,[aux]);assert count['segmentation']==1
    assert torch.isfinite(loss['segmentation'])

def test_missing_metric_labels_are_excluded_not_nan():
    rows=[{'patient_id':'a','rmse':1.,'pcr_brier':.1},{'patient_id':'b','rmse':2.}]
    result=summarize_rows(rows)
    assert result['pcr_brier']['patient_macro_mean']==.1

def test_unknown_config_errors(cfg):
    d=cfg.to_dict();d['loss']['made_up']=1
    with pytest.raises(ValueError):from_dict(d)

def test_source_reader_never_loads_target(store):
    pair=store.records('test')[0]
    seen=[];original=store.read_raw
    def reader(k):seen.append(k);return original(k)
    store.read_raw=reader
    store.source_batch(pair)
    assert seen==[pair['source']]
