#!/usr/bin/env python3
"""Build a full upstream+world_v2 ZIP from a local checkout or an explicit clone.

Uses git archive on a pinned public commit, NEVER copies untracked patient data,
local .env files, checkpoints, cached arrays, or private Git credentials.
"""
from pathlib import Path
import argparse
import io
import subprocess
import tempfile
import zipfile
from install_into_repo import install

UPSTREAM="https://github.com/cgznb/symm-fm.git"
COMMIT="92265d3b1749ae3b686f2089843c49da129fd4d2"


def assemble(output,local_upstream=None,clone=False):
    out=Path(output).resolve()
    if out.exists(): raise FileExistsError(out)
    if out.suffix!=".zip":raise ValueError("Output must end with .zip")
    with tempfile.TemporaryDirectory(prefix="symm_world_assemble_") as tmp:
        work=Path(tmp)
        if local_upstream:
            source=Path(local_upstream).resolve()
        elif clone:
            source=work/"upstream_git"
            subprocess.run(["git","clone","--no-checkout",UPSTREAM,str(source)],check=True)
        else:
            raise ValueError("Supply --local-upstream or explicitly permit --clone")
        resolved=subprocess.check_output(["git","-C",str(source),"rev-parse",COMMIT+"^{commit}"],text=True).strip()
        if resolved!=COMMIT: raise ValueError("Pinned upstream commit mismatch")
        archive=subprocess.check_output(["git","-C",str(source),"archive","--format=zip",COMMIT])
        repo=work/"symm-fm-world-v2-full";repo.mkdir()
        with zipfile.ZipFile(io.BytesIO(archive)) as z:
            for item in z.infolist():
                target=(repo/item.filename).resolve()
                if not target.is_relative_to(repo):raise ValueError("Unsafe archive path")
                if (item.external_attr>>16)&0o170000==0o120000:raise ValueError("Symlinks in source archive are not copied")
            z.extractall(repo)
        install(repo)
        (repo/"WORLD_V2_UPSTREAM_COMMIT.txt").write_text(COMMIT+"\n")
        out.parent.mkdir(parents=True,exist_ok=True)
        with zipfile.ZipFile(out,"w",zipfile.ZIP_DEFLATED,compresslevel=6) as z:
            for f in sorted(repo.rglob("*")):
                if f.is_file():z.write(f,f.relative_to(work))
    return out

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output",required=True)
    g=p.add_mutually_exclusive_group(required=True);g.add_argument("--local-upstream");g.add_argument("--clone",action="store_true")
    a=p.parse_args();print(assemble(a.output,a.local_upstream,a.clone))
