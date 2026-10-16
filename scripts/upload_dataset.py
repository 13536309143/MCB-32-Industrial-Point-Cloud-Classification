"""Upload a prepared release using an existing Hugging Face login; never store tokens."""
import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import HfApi, get_token

ROOT=Path(__file__).resolve().parents[1]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--folder',type=Path,default=ROOT/'.release/huggingface')
    p.add_argument('--repo-id',help='namespace/repository; default: authenticated user/MCB-32-PointCloud')
    p.add_argument('--public',action='store_true',help='Explicitly create a public repository; otherwise private')
    a=p.parse_args()
    if not get_token():raise SystemExit('Run hf auth login with a write-capable token, then retry. Do not put tokens in source files.')
    manifest=json.loads((a.folder/'release_manifest.json').read_text())
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
    api.upload_folder(repo_id=repo,repo_type='dataset',folder_path=a.folder,
                      ignore_patterns=['.cache/**','*.partial'],commit_message='Publish verified MCB-32 nine-representation point cloud dataset')
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
                 archives_verified=len(manifest['archives']),npy_files=manifest['npy_files'],public=not remote.private)
    receipt_path=ROOT/'.release/huggingface_upload_receipt.json'
    receipt_path.parent.mkdir(exist_ok=True);receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt,indent=2),flush=True)


if __name__=='__main__':main()
