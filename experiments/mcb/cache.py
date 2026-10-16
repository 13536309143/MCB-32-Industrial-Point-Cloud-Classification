"""Pack small NPY files into local mmap arrays, keeping all point counts and row order."""
import os
from pathlib import Path
import numpy as np
from .common import DATASETS, read_json, sha_file, write_json


def pack_data(root, index, cache):
    cache = Path(cache) / index['data_sha256'][:20]
    cache.mkdir(parents=True, exist_ok=True)
    records = index['records']
    for variant in DATASETS:
        n = int(variant.rsplit('_', 1)[1])
        target = cache / f'{variant}.npy'
        meta = cache / f'{variant}.json'
        if target.is_file() and meta.is_file():
            old = read_json(meta)
            st = target.stat()
            if (old.get('data_sha256') == index['data_sha256'] and old.get('split_sha256') == index['split_sha256']
                    and old.get('size') == st.st_size and old.get('mtime_ns') == st.st_mtime_ns):
                print('Packed data cached:', variant, flush=True)
                continue
        tmp = target.with_suffix('.tmp.npy')
        array = np.lib.format.open_memmap(tmp, mode='w+', dtype=np.float32, shape=(len(records), n, 3))
        for i, record in enumerate(records):
            array[i] = np.load(Path(root) / variant / record['relative'], allow_pickle=False)
            if i % 5000 == 0:
                print(f'Packing {variant}: {i:,}/{len(records):,}', flush=True)
        array.flush()
        del array
        os.replace(tmp, target)
        st = target.stat()
        write_json(meta, dict(data_sha256=index['data_sha256'], split_sha256=index['split_sha256'],
                             size=st.st_size, mtime_ns=st.st_mtime_ns, sha256=sha_file(target)))
    write_json(cache / 'rows.json', {r['id']: i for i, r in enumerate(records)})
    return cache
