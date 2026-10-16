"""Package byte-identical NPY clouds as deterministic, checksummed TAR shards."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import sqlite3
import tarfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sha_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=ROOT/'data/MCB_pointcloud')
    p.add_argument('--index-dir',type=Path,default=ROOT/'results/key6/_data')
    p.add_argument('--output',type=Path,default=ROOT/'.release/huggingface')
    p.add_argument('--shard-mib',type=int,default=512)
    args=p.parse_args()
    if args.shard_mib<1: p.error('--shard-mib must be positive')
    index=json.loads((args.index_dir/'index.json').read_text())
    output=args.output.resolve();output.mkdir(parents=True,exist_ok=True)
    done=output/'release_manifest.json'
    if done.exists():
        manifest=json.loads(done.read_text())
        if manifest['data_sha256']!=index['data_sha256']:
            raise RuntimeError('Existing release binds a different dataset; use a new output directory')
        for row in manifest['archives']:
            path=output/row['path']
            if not path.is_file() or path.stat().st_size!=row['bytes'] or sha_file(path)!=row['sha256']:
                raise RuntimeError(f"Existing archive failed verification: {row['path']}")
        print('Existing release verified:',output,flush=True)
        return
    meta=output/'metadata';meta.mkdir(exist_ok=True)
    for name in ['README.md','README.zh-CN.md']:
        shutil.copy2(ROOT/'huggingface'/name,output/name)
    if (ROOT/'huggingface/assets').is_dir():
        shutil.copytree(ROOT/'huggingface/assets',output/'assets',dirs_exist_ok=True)
    shutil.copy2(ROOT/'THIRD_PARTY_NOTICES.md',output/'THIRD_PARTY_NOTICES.md')
    for source,target in [('class_to_idx.json','class_to_idx.json')]:
        shutil.copy2(args.index_dir/source,meta/target)
    with (args.index_dir/'splits.csv').open('rb') as src,gzip.open(meta/'experiment_splits.csv.gz','wb') as dst:
        shutil.copyfileobj(src,dst)
    # The independent inventory ties this release to the data used in the experiment.
    db=sqlite3.connect(f'file:{args.index_dir / "inventory.sqlite3"}?mode=ro',uri=True)
    expected={r[0]:(r[1],r[2]) for r in db.execute('SELECT path,size,sha FROM files')}
    db.close()
    expected_total=len(index['records'])*len(index['datasets'])
    if len(expected)!=expected_total:
        raise RuntimeError('Incomplete reference inventory')
    archives=[];count=0;logical_bytes=0;classes=[]
    for label,name in enumerate(index['classes']):
        group=[r for r in index['records'] if r['label']==label]
        classes.append(dict(index=label,name=name,original_train=sum(r['relative'].startswith('train/') for r in group),
                            original_test=sum(r['relative'].startswith('test/') for r in group),
                            experiment_train=sum(r['split']=='train' for r in group),
                            experiment_val=sum(r['split']=='val' for r in group)))
    (meta/'class_counts.json').write_text(json.dumps(classes,indent=2)+'\n')
    with gzip.open(meta/'file_manifest.csv.gz','wt',encoding='utf-8',newline='') as gz:
        writer=csv.writer(gz);writer.writerow(['path','bytes','sha256'])
        for variant in index['datasets']:
            n=int(variant.rsplit('_',1)[1])
            for split in ['train','test']:
                records=sorted((r for r in index['records'] if r['relative'].startswith(split+'/')),key=lambda r:r['relative'])
                archive=None;tmp=None;archive_size=0;shard=-1;shard_count=0
                def close_archive():
                    nonlocal archive
                    if archive is None:return
                    archive.close();final=tmp.with_suffix('');os.replace(tmp,final)
                    row=dict(path=final.relative_to(output).as_posix(),bytes=final.stat().st_size,
                             sha256=sha_file(final),files=shard_count,dataset=variant,original_split=split)
                    archives.append(row)
                    print(f"{row['path']}: {row['files']} files; checked {count:,}/{expected_total:,}",flush=True)
                    archive=None
                for r in records:
                    rel=f"{variant}/{r['relative']}";path=args.data/rel
                    size=path.stat().st_size
                    framed=512+((size+511)//512)*512
                    if archive is None or (archive_size+framed>args.shard_mib*1024**2 and shard_count):
                        close_archive();shard+=1;archive_size=0;shard_count=0
                        final=output/'archives'/variant/f'{split}-{shard:05d}.tar'
                        final.parent.mkdir(parents=True,exist_ok=True)
                        tmp=final.with_suffix('.tar.partial')
                        archive=tarfile.open(tmp,'w',format=tarfile.USTAR_FORMAT)
                    raw=path.read_bytes();actual=hashlib.sha256(raw).hexdigest()
                    if expected.get(rel)!=(len(raw),actual):
                        raise RuntimeError(f'Data differs from experimental inventory: {rel}')
                    xyz=np.load(io.BytesIO(raw),allow_pickle=False)
                    if xyz.shape!=(n,3) or xyz.dtype!=np.float32 or not xyz.flags.c_contiguous or not np.isfinite(xyz).all():
                        raise RuntimeError(f'Invalid NPY: {rel}')
                    info=tarfile.TarInfo('MCB_pointcloud/'+rel);info.size=len(raw)
                    info.mode=0o644;info.mtime=0;info.uid=info.gid=0
                    archive.addfile(info,io.BytesIO(raw))
                    writer.writerow([info.name,len(raw),actual])
                    count+=1;logical_bytes+=len(raw);archive_size+=framed;shard_count+=1
                close_archive()
    assert count==expected_total
    manifest=dict(format_version=1,format='uncompressed USTAR shards; original NPY bytes',
                  objects=len(index['records']),classes=len(index['classes']),npy_files=count,
                  npy_logical_bytes=logical_bytes,data_sha256=index['data_sha256'],
                  split_sha256=index['split_sha256'],archives=archives)
    done.write_text(json.dumps(manifest,indent=2)+'\n')
    with (output/'SHA256SUMS').open('w') as f:
        for path in sorted(output.rglob('*')):
            if path.is_file() and path.name!='SHA256SUMS':
                f.write(f'{sha_file(path)}  {path.relative_to(output).as_posix()}\n')
    print(f'Release ready: {count:,} NPYs, {len(archives)} archives, {logical_bytes:,} payload bytes',flush=True)


if __name__=='__main__':main()
