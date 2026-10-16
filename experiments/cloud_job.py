"""Resume the existing S checkpoint, then finish the six-model comparison queue."""
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path
from experiments.mcb.common import ROOT, read_json, sha_file, write_json
from experiments.mcb.suite import environment


def main():
    if sys.platform.startswith('linux'):
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        target = 65536 if hard == resource.RLIM_INFINITY else min(65536, hard)
        resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, target), hard))
    profile = 'key6'
    logs = ROOT / 'remote_logs'
    state_path = logs / 'job_state.json'

    def state(phase, **extra):
        write_json(state_path, dict(phase=phase, profile=profile, planned_training_runs=6,
                                    updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), **extra))
        print('Cloud phase:', phase, flush=True)

    state('verifying_first_dataset')
    expected = read_json(logs / 'pilot_expected.json')
    first = ROOT / 'data/MCB_pointcloud/area_random_2048'
    actual = {p.relative_to(ROOT / 'data/MCB_pointcloud').as_posix() for p in first.rglob('*.npy')}
    if actual != set(expected):
        raise RuntimeError('First dataset upload is incomplete')
    for ordinal, (relative, value) in enumerate(expected.items(), 1):
        path = ROOT / 'data/MCB_pointcloud' / relative
        if path.stat().st_size != value['size'] or sha_file(path) != value['sha256']:
            raise RuntimeError('First dataset checksum mismatch: ' + relative)
        if ordinal % 10000 == 0:
            print('Pilot data SHA check:', ordinal, flush=True)
    write_json(logs / 'pilot_data_verified.json', dict(files=len(expected), all_sha256_matched=True))
    write_json(logs / 'pilot_environment.json', environment())
    run = read_json(logs / 'pilot_run.json')
    folder = ROOT / f'results/{profile}/runs' / run['id']
    state('first_full_data_training', run_id=run['id'], model=run['config']['model'], epochs=run['config']['epochs'])
    command = [sys.executable, '-u', '-m', 'experiments.mcb.train', '--run', str(logs / 'pilot_run.json'),
               '--index', str(logs / 'cloud_index.json'), '--data', str(ROOT / 'data/MCB_pointcloud'), '--output', str(folder)]
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'console.log').open('a', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding='utf-8', errors='replace', bufsize=1)
        for line in process.stdout:
            print(line, end='', flush=True)
            log.write(line)
            log.flush()
        code = process.wait()
    if code:
        state('pilot_failed', exit_code=code)
        raise SystemExit(code)
    state('waiting_for_remaining_upload', first_run_training_complete=True)
    while not (logs / 'upload_complete.json').exists():
        time.sleep(30)
    # Standard suite verifies every byte/ID, reuses the identical first-run checkpoint,
    # completes its cross-representation tests, and launches the remaining five runs.
    state('validating_all_data_and_running_key6_queue')
    code = subprocess.call(['bash', str(ROOT / 'experiments/run_cloud.sh'), profile], cwd=ROOT)
    state('complete' if code == 0 else 'queue_failed', exit_code=code)
    raise SystemExit(code)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
        write_json(ROOT / 'remote_logs/job_state.json', dict(phase='failed', error='See remote_logs/quick.log'))
        raise
