"""Upload a prepared release using an existing Hugging Face login; never store tokens."""
import argparse
import hashlib
import json
import time
from pathlib import Path

from huggingface_hub import HfApi, get_token, CommitOperationAdd
from huggingface_hub.utils import logging as hub_logging

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,default=ROOT/'.release/huggingface')
    p.add_argument('--repo-id',help='namespace/repository; default: authenticated user/MCB-32-PointCloud')
    p.add_argument('--public',action='store_true',help='Explicitly create a public repository; otherwise private')
    p.add_argument('--manifest',default='release_manifest.json',help='Manifest basename, including optional mesh_release_manifest.json')
    p.add_argument('--receipt',type=Path,default=ROOT/'.release/huggingface_upload_receipt.json')
    a=p.parse_args()
    if not get_token():raise SystemExit('Run hf auth login with a write-capable token, then retry. Do not put tokens in source files.')
    manifest=json.loads((a.folder/a.manifest).read_text())
    for item in manifest['archives']:
        path=a.folder/item['path'];h=hashlib.sha256()
        with path.open('rb') as f:
            for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
        if path.stat().st_size!=item['bytes'] or h.hexdigest()!=item['sha256']:
            raise SystemExit(f"Archive verification failed: {item['path']}")
    api=HfApi();user=api.whoami();repo=a.repo_id or f"{user['name']}/MCB-32-PointCloud"
    api.create_repo(repo,repo_type='dataset',private=not a.public,exist_ok=True)
    info=api.repo_info(repo,repo_type='dataset')
    if bool(info.private)==a.public:
        raise SystemExit('Existing repository visibility differs from requested visibility; inspect its settings before continuing.')
    print('Uploading dataset:',repo,flush=True)
    # Emit native periodic summaries even when stdout/stderr are redirected.
    hub_logging.set_verbosity_info()
    # Commit one archive at a time so completed shards survive connection failures.
    present={x.rfilename:x for x in api.repo_info(repo,repo_type='dataset',files_metadata=True).siblings}
    for number,item in enumerate(manifest['archives'],1):
        old=present.get(item['path'])
        if old and old.size==item['bytes'] and old.lfs and old.lfs.sha256==item['sha256']:
            print(f"Verified existing {number}/{len(manifest['archives'])}: {item['path']}",flush=True)
            continue
        for attempt in range(1,5):
            try:
                api.create_commit(repo_id=repo,repo_type='dataset',operations=[CommitOperationAdd(path_in_repo=item['path'],path_or_fileobj=a.folder/item['path'])],commit_message=f"Publish verified shard {number}/{len(manifest['archives'])}")
                print(f"Committed {number}/{len(manifest['archives'])}: {item['path']}",flush=True)
                break
            except Exception as exc:
                print(f"Attempt {attempt} failed: {type(exc).__name__}: {exc}",flush=True)
                if attempt==4:raise
                time.sleep(10)
    metadata=[x for x in sorted(a.folder.rglob('*')) if x.is_file() and '.cache' not in x.parts and not x.name.endswith('.partial') and x.relative_to(a.folder).as_posix() not in {r['path'] for r in manifest['archives']}]
    api.create_commit(repo_id=repo,repo_type='dataset',operations=[CommitOperationAdd(path_in_repo=x.relative_to(a.folder).as_posix(),path_or_fileobj=x) for x in metadata],commit_message='Publish point cloud manifests and provenance')
    remote=api.repo_info(repo,repo_type='dataset',files_metadata=True)
    files={s.rfilename:s for s in remote.siblings}
    for item in manifest['archives']:
        entry=files.get(item['path'])
        if entry is None or entry.size!=item['bytes']:
            raise SystemExit(f"Remote size/path mismatch: {item['path']}")
        if entry.lfs and entry.lfs.sha256!=item['sha256']:
            raise SystemExit(f"Remote SHA-256 mismatch: {item['path']}")
        if not entry.lfs:
            raise SystemExit(f"Missing remote large-file hash: {item['path']}")
    receipt=dict(repo_id=repo,url=f'https://huggingface.co/datasets/{repo}',revision=remote.sha,
                 archives_verified=len(manifest['archives']),npy_files=manifest.get('npy_files',0),obj_files=manifest.get('obj_files',0),public=not remote.private)
    receipt_path=a.receipt
    receipt_path.parent.mkdir(exist_ok=True);receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
