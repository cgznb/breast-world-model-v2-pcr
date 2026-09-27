import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from symm_world.data import DatasetStore
from symm_world.utils import write_json
from symm_world.auxiliary import build_auxiliary
from symm_world.teachers import export_teacher_targets


def with_images(store,tmp_path):
    d=copy.deepcopy(store.manifest);d['image_normalization']={'mean':0.,'std':1.}
    for i,v in enumerate(d['views']):
        for k in ('latent','auxiliary'):v[k]=str(store.resolve(v[k]))
        x=np.stack([np.ones((4,8,8),np.float32)*k for k in (1,3,4)])
        support=np.ones_like(x,dtype=bool);support[:,0,0,0]=False
        x[:,0,0,0]=1e6
        path=tmp_path/(v['id']+'_images.npz')
        np.savez_compressed(path,images=x,support=np.packbits(support.reshape(-1)),tumor_mask=np.zeros((1,4,8,8),np.float32))
        v['images']=str(path)
    path=tmp_path/'image_manifest.json';write_json(path,d);return DatasetStore(path)


def test_measured_kinetics_mask_and_other_aux_preserved(store,tmp_path):
    s=with_images(store,tmp_path)
    path=tmp_path/'measured.json'
    counts=build_auxiliary(s,path,tmp_path/'aux',grid=(1,1,1))
    assert counts['measured']==21
    aux=DatasetStore(path).read_aux(next(iter(s.views)),(1,1,1))
    torch.testing.assert_close(aux['kinetics'].flatten(),torch.tensor([2.,3.,1.]))
    assert torch.equal(aux['segmentation'],torch.zeros_like(aux['segmentation']))


def test_measured_kinetics_refuses_unknown_normalization(store,tmp_path):
    with pytest.raises(ValueError,match='normalization'):
        build_auxiliary(store,tmp_path/'new.json',tmp_path/'aux')


def test_teacher_actual_artifact_and_dimension_validation(store,tmp_path,cfg):
    s=with_images(store,tmp_path)
    teacher=torch.nn.Conv3d(1,32,3,padding=1)
    ts=torch.jit.trace(teacher,torch.ones(3,1,4,8,8));path=tmp_path/'trusted_fixture_teacher.pt';ts.save(str(path))
    out=tmp_path/'teacher_manifest.json'
    report=export_teacher_targets(s,path,out,tmp_path/'teacher_aux',cfg)
    assert report['exported_views']==21
    aux=DatasetStore(out).read_aux(next(iter(s.views)),cfg.encoder.token_grid)
    assert aux['external_tokens'].shape==(4,32)
    assert 'kinetics' in aux
    cfg.encoder.dim=48
    with pytest.raises(ValueError,match='dimension'):
        export_teacher_targets(s,path,tmp_path/'bad.json',tmp_path/'bad_aux',cfg)


def test_additive_installer_preserves_original_and_refuses_overwrite(tmp_path):
    root=Path(__file__).resolve().parents[1]
    spec=importlib.util.spec_from_file_location('installer',root/'scripts/install_into_repo.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    repo=tmp_path/'upstream';(repo/'src/ispy2_symmflow').mkdir(parents=True)
    original=repo/'README.md';original.write_text('untouched original')
    target=module.install(repo)
    assert (target/'src/symm_world/cli.py').exists() and (repo/'run_world_v2.py').exists()
    assert original.read_text()=='untouched original'
    with pytest.raises(FileExistsError):module.install(repo)


def test_full_assembler_uses_only_tracked_source(tmp_path,monkeypatch):
    import sys,subprocess,zipfile
    root=Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'scripts'))
    import assemble_full_repository as assembler
    repo=tmp_path/'fixture_git';(repo/'src/ispy2_symmflow').mkdir(parents=True)
    (repo/'src/ispy2_symmflow/__init__.py').write_text('# tracked fixture\n')
    (repo/'README.md').write_text('original tracked source')
    subprocess.run(['git','init','-q',str(repo)],check=True)
    subprocess.run(['git','-C',str(repo),'add','.'],check=True)
    subprocess.run(['git','-C',str(repo),'-c','user.name=Unit test','-c','user.email=test@example.invalid','commit','-qm','fixture'],check=True)
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    (repo/'patient.npz').write_bytes(b'private-untracked-fixture')
    (repo/'.env').write_text('never archive this untracked fixture')
    monkeypatch.setattr(assembler,'COMMIT',commit)
    output=tmp_path/'full.zip';assembler.assemble(output,local_upstream=repo)
    with zipfile.ZipFile(output) as z:
        names=z.namelist()
        assert any(n.endswith('run_world_v2.py') for n in names)
        assert any(n.endswith('src/symm_world/encoder.py') for n in names)
        assert not any(n.endswith(('patient.npz','.env')) for n in names)
