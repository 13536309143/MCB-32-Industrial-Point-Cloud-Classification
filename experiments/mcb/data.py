import argparse
import csv
import os
import multiprocessing
import sqlite3
from pathlib import Path
import numpy as np
from torch.utils.data import Dataset
from .common import DATASETS, digest, read_json, seed_for, sha_file, write_json


def prepare_data(root, output, expected_count=50531, val_fraction=0.1, split_seed=20261013):
    if not 0 < val_fraction < 1:
        raise ValueError('Validation fraction must be strictly between 0 and 1')
    root, output = Path(root), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    first = root / DATASETS[0]
    classes = sorted(p.name for p in (first / 'train').iterdir() if p.is_dir())
    if len(classes) != 32 and expected_count:
        raise ValueError(f'Expected 32 classes, found {len(classes)}')
    canonical = {}
    for split in ('train', 'test'):
        for label, cls in enumerate(classes):
            paths = sorted((first / split / cls).glob('*.npy'))
            if not paths:
                raise ValueError(f'Empty class: {split}/{cls}')
            for path in paths:
                model_id = path.stem
                if model_id in canonical:
                    raise ValueError(f'Model ID repeated across classes or splits: {model_id}')
                canonical[model_id] = dict(id=model_id, label=label, cls=cls, split=split,
                                           relative=f'{split}/{cls}/{path.name}')
    if expected_count and len(canonical) != expected_count:
        raise ValueError(f'Expected {expected_count} models, found {len(canonical)}')
    db = sqlite3.connect(output / 'inventory.sqlite3')
    db.execute('CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, size INTEGER, mtime INTEGER, sha TEXT)')
    inventory_hashes, count, cached = [], 0, 0
    try:
        for variant in DATASETS:
            n = int(variant.rsplit('_', 1)[1])
            folder = root / variant
            seen = set()
            for split in ('train', 'test'):
                actual_classes = sorted(p.name for p in (folder / split).iterdir() if p.is_dir())
                if actual_classes != classes:
                    raise ValueError(f'Class directories disagree: {variant}/{split}')
                for cls in classes:
                    for path in sorted((folder / split / cls).glob('*.npy')):
                        rec = canonical.get(path.stem)
                        rel = path.relative_to(root).as_posix()
                        if rec is None or rec['relative'] != path.relative_to(folder).as_posix() or path.stem in seen:
                            raise ValueError(f'Identity/label/split mismatch: {rel}')
                        seen.add(path.stem)
                        st = path.stat()
                        old = db.execute('SELECT size, mtime, sha FROM files WHERE path=?', (rel,)).fetchone()
                        if old and old[:2] == (st.st_size, st.st_mtime_ns):
                            sha = old[2]
                            cached += 1
                        else:
                            points = np.load(path, allow_pickle=False)
                            if points.dtype != np.float32 or points.shape != (n, 3) or not points.flags.c_contiguous:
                                raise ValueError(f'Invalid float32 XYZ shape/layout: {rel}')
                            if not np.isfinite(points).all() or np.linalg.norm(points.astype(np.float64), axis=1).max() > 1.00001:
                                raise ValueError(f'Invalid coordinates/bounds: {rel}')
                            sha = sha_file(path)
                            db.execute('INSERT OR REPLACE INTO files VALUES (?,?,?,?)', (rel, st.st_size, st.st_mtime_ns, sha))
                        inventory_hashes.append((rel, sha))
                        count += 1
                        if count % 5000 == 0:
                            db.commit()
                            print(f'Data check {count:,}/{len(canonical)*9:,}; cached {cached:,}', flush=True)
            if seen != set(canonical):
                raise ValueError(f'Missing {len(set(canonical)-seen)} models in {variant}')
        db.commit()
    finally:
        db.close()
    records = list(canonical.values())
    for label in range(len(classes)):
        train = [r for r in records if r['label'] == label and r['split'] == 'train']
        train.sort(key=lambda r: seed_for('validation', split_seed, r['id']))
        count_val = min(len(train)-1, max(1, round(len(train)*val_fraction)))
        if count_val < 1:
            raise ValueError('Each class needs at least two original training models')
        for r in train[:count_val]:
            r['split'] = 'val'
    records.sort(key=lambda r: (r['label'], r['id']))
    value = dict(classes=classes, class_to_idx={c: i for i, c in enumerate(classes)}, records=records,
                 datasets=list(DATASETS), data_sha256=digest(inventory_hashes),
                 split_seed=split_seed, val_fraction=val_fraction, checked_files=count,
                 check_cache='size + nanosecond mtime; delete inventory.sqlite3 to force rehash')
    value['split_sha256'] = digest([(r['id'], r['label'], r['split']) for r in records])
    value['identity_sha256'] = digest([(r['id'], r['label'], r['relative']) for r in records])
    write_json(output / 'index.json', value)
    write_json(output / 'class_to_idx.json', value['class_to_idx'])
    with (output / 'splits.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=('id', 'label', 'cls', 'split', 'relative'))
        writer.writeheader()
        writer.writerows(records)
    return value


def selected_records(index, split, fraction=1.0):
    if not 0 < fraction <= 1:
        raise ValueError('Training fraction must be in (0, 1]')
    records = [r for r in index['records'] if r['split'] == split]
    if fraction < 1:
        selected = []
        for label in range(len(index['classes'])):
            group = [r for r in records if r['label'] == label]
            group.sort(key=lambda r: seed_for('fraction', index['split_seed'], r['id']))
            selected.extend(group[:max(1, round(len(group)*fraction))])
        records = selected
    return records


def rotation_matrix(rng, angle=None):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    if angle is None:
        # Haar-uniform SO(3), via a unit quaternion.
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        w, x, y, z = q
        return np.array([[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                         [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                         [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]], dtype=np.float32)
    x, y, z = axis
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return (np.eye(3)*np.cos(angle) + (1-np.cos(angle))*np.outer(axis, axis) + np.sin(angle)*cross).astype(np.float32)


def corrupt(x, kind, severity, rng):
    x = x.copy()
    s = max(0, severity-1)
    if kind == 'jitter':
        sigma = (0.005, 0.01, 0.02)[s]
        x += np.clip(rng.normal(0, sigma, x.shape), -3*sigma, 3*sigma).astype(np.float32)
    elif kind == 'rotation':
        x = x @ rotation_matrix(rng, np.deg2rad((15, 45, 90)[s])).T
    elif kind == 'so3':
        x = x @ rotation_matrix(rng).T
    elif kind == 'scale':
        factor = (1.15, 1.3, 1.5)[s]
        x *= factor if rng.random() < 0.5 else 1/factor
    elif kind == 'translation':
        direction = rng.normal(size=3)
        x += (direction / np.linalg.norm(direction)*(0.05, 0.1, 0.2)[s]).astype(np.float32)
    elif kind in ('dropout', 'occlusion'):
        keep = max(1, round(len(x)*(1-(0.1, 0.3, 0.5)[s])))
        if kind == 'dropout':
            idx = rng.choice(len(x), keep, replace=False)
        else:
            direction = rng.normal(size=3)
            idx = np.argsort(x @ direction)[:keep]  # planar crop approximation
        remaining = x[idx]
        extra = rng.choice(keep, len(x)-keep, replace=True)
        x = np.concatenate((remaining, remaining[extra]))
        rng.shuffle(x)
    elif kind == 'outliers':
        idx = rng.choice(len(x), round(len(x)*(0.01, 0.05, 0.1)[s]), replace=False)
        x[idx] = rng.uniform(-1, 1, (len(idx), 3)).astype(np.float32)
    elif kind == 'permutation':
        rng.shuffle(x)
    elif kind != 'clean':
        raise ValueError(kind)
    return np.ascontiguousarray(x, dtype=np.float32)


class Clouds(Dataset):
    def __init__(self, root, variant, records, seed=42, augmentation='none', corruption='clean', severity=0):
        self.root = Path(root) / variant
        self.variant, self.records = variant, records
        self.seed, self.augmentation = seed, augmentation
        self.corruption, self.severity = corruption, severity
        self.epoch_shared = multiprocessing.Value('i', 0)
        self.packed = None
        pack_root = os.environ.get('MCB_PACK_ROOT')
        self.pack_path = Path(pack_root) / f'{variant}.npy' if pack_root else None
        self.pack_rows = read_json(Path(pack_root) / 'rows.json') if pack_root else None

    @property
    def epoch(self):
        return self.epoch_shared.value

    @epoch.setter
    def epoch(self, value):
        self.epoch_shared.value = value

    def __getstate__(self):
        value = self.__dict__.copy()
        value['packed'] = None
        return value

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item):
        r = self.records[item]
        if self.pack_path is not None:
            if self.packed is None:
                self.packed = np.load(self.pack_path, mmap_mode='r', allow_pickle=False)
            x = self.packed[self.pack_rows[r['id']]].copy()
        else:
            x = np.load(self.root / r['relative'], allow_pickle=False).copy()
        # Per-model random streams independent of worker count and batch size.
        rng = np.random.default_rng(seed_for(self.seed, r['id'], self.epoch, self.augmentation,
                                            self.corruption, self.severity))
        if self.augmentation != 'none':
            rng.shuffle(x)
            if self.augmentation in ('so3', 'strong'):
                x = x @ rotation_matrix(rng).T
            elif self.augmentation == 'classic':
                angle = rng.uniform(-np.pi, np.pi)
                c, s = np.cos(angle), np.sin(angle)
                x = x @ np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float32).T
            x *= rng.uniform(0.8, 1.25, 3).astype(np.float32)
            x += rng.uniform(-0.1, 0.1, 3).astype(np.float32)
            if self.augmentation in ('classic', 'strong'):
                x = corrupt(x, 'jitter', 2, rng)
            if self.augmentation == 'strong':
                x = corrupt(x, 'dropout', 2, rng)
        x = corrupt(x, self.corruption, self.severity, rng)
        return x, r['label'], r['id']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    value = prepare_data(args.root, args.output)
    print({s: len(selected_records(value, s)) for s in ('train', 'val', 'test')})


if __name__ == '__main__':
    main()
