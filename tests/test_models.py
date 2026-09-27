from dataclasses import asdict
import pytest
import torch
from symm_world.layers import SwinBlock3D,partition_windows,reverse_windows
from symm_world.encoder import corrupt_latent
from symm_world.models import WorldModel
from symm_world.losses import representation_loss,pair_flow_loss,rollout_loss
from symm_world.codec import MRILevelVQGAN,VQConfig,FrozenThreePhaseCodec,load_codec
import copy


def test_window_roundtrip():
    x=torch.randn(2,4,6,8,12);w=(2,3,4)
    y=partition_windows(x,w)
    z=reverse_windows(y,w,(4,6,8),2)
    torch.testing.assert_close(z,x)

@pytest.mark.parametrize("shape",[(3,5,7),(1,2,2),(4,4,4)])
def test_shifted_window_odd_padding_backward(shape):
    # Real shifted-window attention, not a dummy backend.
    block=SwinBlock3D(16,4,(2,4,4),shifted=True)
    x=torch.randn(1,16,*shape,requires_grad=True)
    y=block(x);assert y.shape==x.shape
    y.square().mean().backward();assert torch.isfinite(x.grad).all() and x.grad.abs().sum()>0

def test_encoder_phase_mixing_and_trainable_layers(rep,store,cfg):
    x=store.normalize(store.read_raw(store.records('train')[0]['source']))[None]
    s=rep.encoder(x);assert s.dense.shape==(1,4,32)
    assert s.anatomy.shape==(1,2,32) and s.disease.shape==(1,2,32)
    swapped=x.reshape(1,3,8,*x.shape[2:])[:,[1,0,2]].reshape_as(x)
    assert not torch.allclose(rep.encoder(swapped).dense,s.dense)
    s.dense.square().mean().backward()
    assert sum(p.grad is not None for p in rep.encoder.parameters())>30

def test_masks_are_applied_before_encoding(cfg):
    x=torch.randn(2,24,4,8,8);out,mask=corrupt_latent(x,cfg.encoder)
    assert mask.any() and out.shape==x.shape
    assert (out==0).any();assert torch.equal(out[out!=0],x[out!=0])

def test_jepa_teacher_has_no_grad_and_updates(rep,store,cfg):
    batch=store.pair_batch(store.records('train')[:1],cfg.encoder.token_grid)
    loss,parts,summary=representation_loss(rep,batch);loss.backward()
    assert parts['jepa']>0 and all(p.grad is None for p in rep.target_encoder.parameters())
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in rep.masked_predictor.parameters())
    before=next(rep.target_encoder.parameters()).clone()
    with torch.no_grad():next(rep.encoder.parameters()).add_(.1)
    rep.update_target(.9);assert not torch.equal(before,next(rep.target_encoder.parameters()))

def test_world_frozen_encoder_propagates_input_gradient(rep,cfg,store):
    world=WorldModel(cfg,rep);x=torch.randn(1,24,4,8,8,requires_grad=True)
    world.encoder(x).disease.square().mean().backward()
    assert x.grad.abs().sum()>0
    assert all(p.grad is None for p in world.encoder.parameters())

def test_rollout_reads_no_true_intermediate_and_has_gradient(rep,cfg,store):
    model=WorldModel(cfg,rep)
    with torch.no_grad():model.velocity.out.weight.normal_(0,.001)
    teacher=copy.deepcopy(model).eval().requires_grad_(False)
    generated=[];sampler=model.sample
    def capture(*a,**k):
        z=sampler(*a,**k);z.retain_grad();generated.append(z);return z
    model.sample=capture
    triple=store.triplets(cfg.rollout)[0][0]
    seen=[];original=store.read_raw
    def reader(key):seen.append(key);return original(key)
    store.read_raw=reader
    loss,parts=rollout_loss(model,teacher,store,triple,device='cpu');loss.backward()
    assert triple[0]['target'] not in seen
    assert triple[1]['source'] not in seen
    assert parts['rollout_samples']==2
    assert generated[0].grad is not None and generated[0].grad.abs().sum()>0
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.velocity.parameters())

def test_codec_checkpoint_and_input_gradient(tmp_path):
    cfg=VQConfig(hidden_channels=4,num_groups=4,n_codes=32,nearest_chunk_size=32)
    original=MRILevelVQGAN(cfg)
    path=tmp_path/'codec.pt'
    torch.save({'schema':'first_post_unregistered_tumor_roi_v1','model_config':asdict(cfg),'codec_state':original.state_dict()},path)
    codec=load_codec(path)
    z=torch.randn(1,24,2,3,4,requires_grad=True)
    books=codec.codec.quantizer.embeddings.clone()
    image=codec.decode(z);assert image.shape==(1,3,8,12,16)
    image.square().mean().backward()
    assert z.grad.abs().sum()>0 and all(p.grad is None for p in codec.parameters())
    codec.train();assert not codec.training
    assert torch.equal(books,codec.codec.quantizer.embeddings)
    encoded=codec.encode(image.detach());assert encoded.shape==z.shape

def test_monai_backend_is_real_optional_dependency(cfg):
    pytest.importorskip('monai',reason='MONAI is not installed in this CPU environment; native full network is tested')
    from symm_world.velocity import build_velocity
    cfg.velocity.backend='monai'
    model=build_velocity(cfg.velocity,cfg.encoder.dim)
    x=torch.randn(1,48,4,8,8);context=torch.randn(1,6,32)
    assert model(x,torch.tensor([.5]),context).shape==x.shape
