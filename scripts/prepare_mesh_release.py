"""Package original OBJ bytes and conversion sources as an optional Hub extension."""
import argparse
import csv
import gzip
import hashlib
import io
import json
import os
import shutil
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(4 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', type=Path, default=ROOT/'data/MCB_merged')
    p.add_argument('--output', type=Path, default=ROOT/'.release/huggingface_meshes')
    p.add_argument('--shard-mib', type=int, default=512)
    a = p.parse_args()
    if a.shard_mib < 1: p.error('--shard-mib must be positive')
    index = json.loads((ROOT/'results/key6/_data/index.json').read_text())
    expected = {str(Path(r['relative']).with_suffix('.obj')) for r in index['records']}
    actual = {x.relative_to(a.source).as_posix() for x in a.source.rglob('*.obj')}
    if actual != expected: raise RuntimeError('OBJ IDs/classes/partitions do not match point clouds')
    out = a.output; out.mkdir(parents=True, exist_ok=True)
    manifest_path = out/'mesh_release_manifest.json'
    if manifest_path.exists():
        for row in json.loads(manifest_path.read_text())['archives']:
            f = out/row['path']
            if f.stat().st_size != row['bytes'] or sha_file(f) != row['sha256']:
                raise RuntimeError(f'Existing shard mismatch: {f}')
        print('Existing mesh release verified', flush=True)
        return
    (out/'metadata').mkdir(exist_ok=True)
    archives = []; total_bytes = 0
    with gzip.GzipFile(filename='', fileobj=(out/'metadata/mesh_file_manifest.csv.gz').open('wb'), mode='wb', mtime=0) as zipped:
        with io.TextIOWrapper(zipped, encoding='utf-8', newline='') as text:
            writer = csv.writer(text); writer.writerow(['path','bytes','sha256'])
            for split in ['train','test']:
                group = sorted(x for x in expected if x.startswith(split+'/'))
                chunks = []; current = []; size = 0
                for rel in group:
                    n = (a.source/rel).stat().st_size
                    if current and size+n > a.shard_mib*1024**2:
                        chunks.append(current); current = []; size = 0
                    current.append(rel); size += n
                if current: chunks.append(current)
                for number, chunk in enumerate(chunks):
                    dest = out/'meshes'/f'{split}-{number:05d}.tar.gz'; dest.parent.mkdir(exist_ok=True)
                    tmp = dest.with_suffix('.gz.partial'); hashes = {}
                    with tmp.open('wb') as f, gzip.GzipFile(filename='', fileobj=f, mode='wb', compresslevel=6, mtime=0) as gz, tarfile.open(fileobj=gz, mode='w|', format=tarfile.USTAR_FORMAT) as tar:
                        for rel in chunk:
                            raw = (a.source/rel).read_bytes(); digest = hashlib.sha256(raw).hexdigest()
                            name = 'MCB_merged/'+rel; hashes[name] = digest
                            info = tarfile.TarInfo(name); info.size = len(raw); info.mode = 0o644; info.mtime = 0
                            tar.addfile(info, io.BytesIO(raw)); writer.writerow([name,len(raw),digest]); total_bytes += len(raw)
                    # Independent read-back of every mesh in the compressed shard.
                    with tarfile.open(tmp,'r:gz') as tar:
                        seen = set()
                        for member in tar:
                            if hashlib.sha256(tar.extractfile(member).read()).hexdigest() != hashes[member.name]:
                                raise RuntimeError(f'Mesh archive mismatch: {member.name}')
                            seen.add(member.name)
                        if seen != set(hashes): raise RuntimeError('Missing/extra mesh archive member')
                    os.replace(tmp,dest)
                    archives.append(dict(path=dest.relative_to(out).as_posix(),bytes=dest.stat().st_size,sha256=sha_file(dest),files=len(chunk),original_split=split))
                    print(f'{dest.name}: {len(chunk)} verified OBJ files, {dest.stat().st_size:,} compressed bytes',flush=True)
    conversion = out/'conversion'; conversion.mkdir(exist_ok=True)
    shutil.copy2(ROOT/'data/convert_mcb_9sets.bat',conversion/'convert_mcb_9sets.bat')
    shutil.copytree(ROOT/'data/_pointcloud_converter',conversion/'_pointcloud_converter',ignore=shutil.ignore_patterns('__pycache__','*.pyc'),dirs_exist_ok=True)
    shutil.copy2(ROOT/'data/README.md',conversion/'README.zh-CN.md')
    shutil.copy2(ROOT/'huggingface/CONVERSION.md',conversion/'README.md')
    manifest = dict(format_version=1,format='deterministic gzip-compressed USTAR; original OBJ bytes',obj_files=len(expected),classes=len(index['classes']),obj_logical_bytes=total_bytes,paired_pointcloud_data_sha256=index['data_sha256'],archives=archives)
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Mesh release ready: {len(expected):,} meshes, {len(archives)} shards',flush=True)


if __name__ == '__main__': main()
