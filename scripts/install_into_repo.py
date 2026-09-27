#!/usr/bin/env python3
"""Install an additive workflow without replacing the user's original files."""
from pathlib import Path
import argparse
import shutil

ROOT=Path(__file__).resolve().parents[1]
ALLOWED=("src","configs","scripts","docs","licenses","tests","reports","world.py","pyproject.toml","README.md","LICENSE","SOURCE_AUDIT.json","RELEASE_MANIFEST.json")


def install(repo):
    repo=Path(repo).resolve()
    if not (repo/"src"/"ispy2_symmflow").is_dir():
        raise ValueError("Not a cgznb/symm-fm checkout")
    target=repo/"workflows"/"world_v2"
    entry=repo/"run_world_v2.py"
    if target.exists() or entry.exists():
        raise FileExistsError("world_v2 already exists; use a fresh checkout or explicitly remove the old additive directory after backing it up")
    target.mkdir(parents=True)
    for name in ALLOWED:
        source=ROOT/name
        if not source.exists(): continue
        if source.is_dir():
            shutil.copytree(source,target/name,ignore=shutil.ignore_patterns("__pycache__","*.pyc",".pytest_cache"))
        else:
            shutil.copy2(source,target/name)
    entry.write_text('from pathlib import Path\nimport sys\nsys.path.insert(0,str(Path(__file__).resolve().parent/"workflows/world_v2/src"))\nfrom symm_world.cli import main\nif __name__=="__main__": main()\n',encoding="utf-8")
    return target


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--repo",required=True)
    a=p.parse_args();print(install(a.repo))
