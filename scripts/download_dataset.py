"""Download verified TAR shards and restore the original NPY directory layout."""
import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath
from huggingface_hub import hf_hub_download


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-id',required=True)
    p.add_argument('--revision',default='main',help='Pin the release commit for reproducible downloads')
    p.add_argument('--variant',action='append',help='Repeat to select variants; default: all nine')
    p.add_argument('--output',type=Path,default=Path('data'))
    a=p.parse_args()
    path=hf_hub_download(a.repo_id,'release_manifest.json',repo_type='dataset',revision=a.revision)
    manifest=json.loads(Path(path).read_text())
    available={r['dataset'] for r in manifest['archives']}
    if a.variant and not set(a.variant)<=available:p.error('Unknown point cloud variant')
    out=a.output.resolve();out.mkdir(parents=True,exist_ok=True)
    for item in manifest['archives']:
        if a.variant and item['dataset'] not in a.variant:continue
        path=Path(hf_hub_download(a.repo_id,item['path'],repo_type='dataset',revision=a.revision))
        h=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
        if h.hexdigest()!=item['sha256'] or path.stat().st_size!=item['bytes']:
            raise RuntimeError(f"Archive check failed: {item['path']}")
        with tarfile.open(path,'r:') as tar:
            members=tar.getmembers()
            for m in members:
                parts=PurePosixPath(m.name).parts
                if not m.isfile() or m.name.startswith('/') or '..' in parts or not parts or parts[0]!='MCB_pointcloud':
                    raise RuntimeError(f'Unsafe archive entry: {m.name}')
            for m in members:
                target=out.joinpath(*PurePosixPath(m.name).parts)
                if not target.resolve().is_relative_to(out):raise RuntimeError('Extraction escapes output directory')
                target.parent.mkdir(parents=True,exist_ok=True)
                with tar.extractfile(m) as src,target.open('wb') as dst:
                    import shutil
                    shutil.copyfileobj(src,dst)
        print('Extracted',item['path'],flush=True)


if __name__=='__main__':main()
