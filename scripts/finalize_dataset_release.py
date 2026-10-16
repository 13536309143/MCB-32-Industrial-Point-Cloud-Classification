"""Finalize bilingual cards and verify point cloud, mesh and conversion Hub files."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path
from huggingface_hub import HfApi, CommitOperationAdd

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4*1024*1024), b''): h.update(b)
    return h.hexdigest()


def local_files(folder):
    return {x.relative_to(folder).as_posix(): x for x in sorted(folder.rglob('*'))
            if x.is_file() and '.cache' not in x.parts and not x.name.endswith('.partial')}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo-id', required=True)
    p.add_argument('--folder', type=Path, default=ROOT/'.release/huggingface')
    p.add_argument('--mesh-folder', type=Path, default=ROOT/'.release/huggingface_meshes')
    a = p.parse_args(); api = HfApi(); folder = a.folder
    manifest = json.loads((folder/'release_manifest.json').read_text())
    meshes = json.loads((a.mesh_folder/'mesh_release_manifest.json').read_text())
    archives = manifest['archives'] + meshes['archives']
    first = api.repo_info(a.repo_id, repo_type='dataset', files_metadata=True)
    remote = {x.rfilename: x for x in first.siblings}
    for item in archives:
        r = remote.get(item['path'])
        if not r or r.size != item['bytes'] or not r.lfs or r.lfs.sha256 != item['sha256']:
            raise SystemExit(f"Finish bulk upload first: {item['path']}")
    changed = []
    for name in ['README.md','README.zh-CN.md']:
        shutil.copy2(ROOT/'huggingface'/name, folder/name); changed.append(name)
    for source in sorted((ROOT/'huggingface/assets').rglob('*')):
        if source.is_file():
            rel = source.relative_to(ROOT/'huggingface'); dest = folder/rel
            dest.parent.mkdir(parents=True,exist_ok=True); shutil.copy2(source,dest); changed.append(rel.as_posix())
    known = {x['path']: x['sha256'] for x in archives}
    files = local_files(folder)
    for name, path in local_files(a.mesh_folder).items():
        if name in files: raise RuntimeError(f'Conflicting release path: {name}')
        files[name] = path
    with (folder/'SHA256SUMS').open('w') as f:
        for name, path in sorted(files.items()):
            if name == 'SHA256SUMS': continue
            f.write(f'{known.get(name) or sha(path)}  {name}\n')
    changed.append('SHA256SUMS')
    commit = api.create_commit(repo_id=a.repo_id, repo_type='dataset',
        operations=[CommitOperationAdd(path_in_repo=n,path_or_fileobj=folder/n) for n in changed],
        commit_message='Finalize bilingual cards, mesh/conversion documentation and complete release checksums')
    final = api.repo_info(a.repo_id, repo_type='dataset', revision=commit.oid, files_metadata=True)
    remote = {x.rfilename:x for x in final.siblings}
    for name, path in files.items():
        entry = remote.get(name)
        if not entry or entry.size != path.stat().st_size: raise RuntimeError(f'Remote size/path differs: {name}')
        if entry.lfs:
            if entry.lfs.sha256 != (known.get(name) or sha(path)): raise RuntimeError(f'Remote SHA-256 differs: {name}')
        else:
            raw = path.read_bytes(); value = hashlib.sha1(f'blob {len(raw)}\0'.encode()+raw).hexdigest()
            if entry.blob_id != value: raise RuntimeError(f'Remote Git blob differs: {name}')
    receipt = dict(repo_id=a.repo_id,url=f'https://huggingface.co/datasets/{a.repo_id}',revision=final.sha,
        public=not final.private,archives_verified=len(archives),pointcloud_archives_verified=len(manifest['archives']),
        mesh_archives_verified=len(meshes['archives']),remote_prepared_files_verified=len(files),
        archive_bytes=sum(x['bytes'] for x in archives),npy_files=manifest['npy_files'],obj_files=meshes['obj_files'],
        unique_objects=manifest['objects'],classes=manifest['classes'],data_sha256=manifest['data_sha256'],
        split_sha256=manifest['split_sha256'],card_viewer_disabled=True,source_meshes_included=True,
        conversion_sources_included=True)
    (ROOT/'.release/huggingface_upload_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2))


if __name__ == '__main__': main()
