"""Preflight, isolated sequential training processes, raw result collection, resume."""
import argparse
import csv
import datetime
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
import torch
from .common import ROOT, digest, read_json, sha_file, write_json
from .data import prepare_data
from .cache import pack_data
from .plan import make_plan


def command_output(command):
    try:
        return subprocess.check_output(command, stderr=subprocess.STDOUT, text=True, timeout=30).strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return str(exc)


def code_fingerprint():
    files = sorted((ROOT / 'experiments').rglob('*.py'))
    files += sorted((ROOT / 'experiments').glob('*.sh'))
    files += sorted((ROOT / 'experiments').glob('*.ps1'))
    files += [ROOT / 'experiments/requirements.txt', ROOT / 'run_experiments.bat']
    files += sorted((ROOT / 'PointNeXt/openpoints/models').rglob('*.py'))
    files += sorted((ROOT / 'PointNeXt/openpoints/cpp/pointnet2_batch/src').glob('*'))
    files += sorted((ROOT / 'PointNeXt/cfgs').rglob('*.yaml'))
    return digest([(p.relative_to(ROOT).as_posix(), sha_file(p)) for p in files if p.is_file()])


class Lock:
    def __init__(self, path):
        self.path = Path(path)
        self.file = None

    def __enter__(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.file = self.path.open('a+b')
        self.file.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                self.file.write(b'0')
                self.file.flush()
                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.file.close()
            raise RuntimeError(f'Another MCB suite is running: {self.path}')
        return self

    def __exit__(self, *args):
        self.file.close()


def environment():
    cuda = torch.cuda.is_available()
    value = dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                 platform=platform.platform(), python=sys.version, executable=sys.executable,
                 torch=torch.__version__, torch_cuda=torch.version.cuda, cuda_available=cuda,
                 packages={d.metadata['Name']: d.version for d in importlib.metadata.distributions()},
                 pointnext_commit=command_output(['git', '-C', str(ROOT / 'PointNeXt'), 'rev-parse', 'HEAD']),
                 openpoints_commit=command_output(['git', '-C', str(ROOT / 'PointNeXt/openpoints'), 'rev-parse', 'HEAD']),
                 nvcc=command_output(['nvcc', '--version']), nvidia_smi=command_output(['nvidia-smi']),
                 code_sha256=code_fingerprint(), cpu_count=os.cpu_count())
    if cuda:
        value.update(gpu=torch.cuda.get_device_name(0), capability=torch.cuda.get_device_capability(0),
                     gpu_total_memory_bytes=torch.cuda.get_device_properties(0).total_memory)
    return value


def collect_raw(output, plan):
    rows = []
    for run in plan['runs']:
        folder = output / 'runs' / run['id']
        training = read_json(folder / 'training_complete.json') if (folder / 'training_complete.json').exists() else {}
        bench = read_json(folder / 'benchmark.json') if (folder / 'benchmark.json').exists() else {}
        for case in run['evaluations']:
            path = folder / 'tests' / f"{case['name']}.json"
            if not path.exists():
                continue
            metrics = read_json(path)
            row = dict(run_id=run['id'], group=run['group'], **run['config'],
                       test_case=case['name'], test_dataset=case['dataset'], corruption=case['corruption'], severity=case['severity'])
            row.update({k: v for k, v in metrics.items() if isinstance(v, (int, float, str))})
            row.update({k: v for k, v in training.items() if isinstance(v, (int, float, str))})
            row.update(latency_batch1_p50_ms=bench.get('batch_1', {}).get('p50_ms'),
                       throughput_batch8_samples_s=bench.get('batch_8', {}).get('throughput_samples_s'),
                       artifact_path=str(folder.relative_to(output)))
            rows.append(row)
    if rows:
        keys = list(dict.fromkeys(k for row in rows for k in row))
        tmp = output / 'results.csv.tmp'
        with tmp.open('w', newline='', encoding='utf-8-sig') as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, output / 'results.csv')


def completed_artifacts_valid(folder):
    try:
        value = read_json(folder / 'complete.json')
        for relative, expected in value['artifacts'].items():
            path = folder / relative
            stat = path.stat()
            if stat.st_size != expected['size']:
                return False
            if stat.st_mtime_ns != expected['mtime_ns'] and sha_file(path) != expected['sha256']:
                return False
        return True
    except (OSError, KeyError, ValueError):
        return False


def estimate_disk_bytes(plan, pack=True):
    params = dict(pointnext_s=1371456, pointnext_b=3879200, pointnext_l=7176416,
                  pointnext_xl=40839968, pointnet=3477225, pointnet2=1470496)
    weights = 0
    for run in plan['runs']:
        cfg = run['config']
        count = params[cfg['model']]
        if cfg['pool'] == 'max_avg':
            count += (1024 if cfg['model'] == 'pointnext_xl' else 512)*512
        # best + last model and optimizer tensors, excluding small metadata.
        weights += count*4*(3 if cfg['optimizer'] == 'sgd' else 4)
    tests = sum(len(r['evaluations']) for r in plan['runs'])*10107*(32*4+64*4+16)
    histories = sum(r['config']['epochs'] for r in plan['runs'])*15000
    return int((weights+tests+histories)*1.1 + (30 if pack else 5)*1024**3)


def run_suite(args):
    started = time.perf_counter()
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with Lock(output / 'suite.lock'), Lock(Path.home() / '.cache/mcb-pointnext/gpu-suite.lock'):
        env = environment()
        if not env['cuda_available'] and not args.allow_cpu:
            raise RuntimeError('CUDA is not available. Actual experiments will never silently run on CPU.')
        free = shutil.disk_usage(output).free
        minimum_free_gib = 5 if args.no_pack else 30
        if free < minimum_free_gib * 1024**3 and not args.allow_cpu:
            raise RuntimeError(f'Less than {minimum_free_gib} GiB free in result location. Change --output to a larger disk.')
        write_json(output / 'environment_latest.json', env)
        old_env_path = output / 'environment_initial.json'
        if old_env_path.exists():
            initial = read_json(old_env_path)
            if initial['code_sha256'] != env['code_sha256'] or initial['torch'] != env['torch']:
                raise RuntimeError('Code or PyTorch changed since this result directory was created. Choose a new directory.')
        else:
            write_json(old_env_path, env)
        plan = read_json(args.plan) if args.plan else make_plan(args.profile)
        plan.pop('plan_sha256', None)
        plan['plan_sha256'] = digest(plan)
        plan_path = output / 'plan.json'
        if plan_path.exists() and read_json(plan_path)['plan_sha256'] != plan['plan_sha256']:
            raise RuntimeError('Experiment plan changed. Choose a new result directory to avoid mixed results.')
        write_json(plan_path, plan)
        print('Plan:', plan['summary'], flush=True)
        split_settings = {(r['config']['split_seed'], r['config']['val_fraction']) for r in plan['runs']}
        if len(split_settings) != 1:
            raise ValueError('All runs must share the same validation split settings.')
        split_seed, val_fraction = next(iter(split_settings))
        index = prepare_data(args.data, output / '_data', expected_count=0 if args.allow_cpu else 50531,
                             split_seed=split_seed, val_fraction=val_fraction)
        estimate = estimate_disk_bytes(plan, pack=not args.no_pack)
        cache_folder = Path(args.cache) / index['data_sha256'][:20]
        cached_bytes = sum(p.stat().st_size for p in cache_folder.glob('*.npy')) if cache_folder.exists() else 0
        required = estimate-min(cached_bytes, 30*1024**3) if not args.no_pack else estimate
        used = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
        write_json(output / 'disk_budget.json', dict(estimated_bytes=estimate, required_after_existing_cache_bytes=required,
                                                   available_bytes=free, existing_result_bytes=used, existing_cache_bytes=cached_bytes,
                                                   method=f'Checkpoint tensors + uncompressed prediction upper estimate + histories + {30 if not args.no_pack else 5} GiB cache/reserve'))
        if not args.allow_cpu and free+used < required:
            raise RuntimeError(f'Insufficient result storage: budget {required/2**30:.1f} GiB; free plus existing results {(free+used)/2**30:.1f} GiB. Use a larger disk or a smaller plan.')
        reference_path = ROOT / 'experiments/data_reference.json'
        if reference_path.exists() and not args.allow_cpu:
            reference = read_json(reference_path)
            if reference['classes'] != index['classes'] or reference['identity_sha256'] != index['identity_sha256']:
                raise RuntimeError('Model identities/labels differ from the verified Mac dataset. Check the copy.')
            if reference['data_sha256'] != index['data_sha256']:
                registry_path = ROOT / 'results/data_reconverted.json'
                registry = read_json(registry_path) if registry_path.exists() else {}
                if getattr(args, 'accept_reconverted_data', False):
                    registry = dict(data_sha256=index['data_sha256'], identity_sha256=index['identity_sha256'],
                                    reason='Successful one-click reconversion; cross-platform floating point variation accepted')
                    write_json(registry_path, registry)
                if registry.get('data_sha256') != index['data_sha256']:
                    raise RuntimeError('Cloud bytes differ from the Mac reference. Check the copy; for an intentional fresh conversion use --accept-reconverted-data in a new result directory.')
                print('Validated reconverted data: model IDs match; the actual point-cloud fingerprint is recorded.', flush=True)
        # Bind completed runs to this dataset as well as to code and plan.
        binding = dict(data_sha256=index['data_sha256'], split_sha256=index['split_sha256'])
        binding_path = output / 'data_binding.json'
        if binding_path.exists() and read_json(binding_path) != binding:
            raise RuntimeError('Dataset changed. Choose a new result directory.')
        write_json(binding_path, binding)
        if not args.no_pack:
            packed = pack_data(args.data, index, args.cache)
            os.environ['MCB_PACK_ROOT'] = str(packed)
        statuses_path = output / 'status.json'
        statuses = read_json(statuses_path) if statuses_path.exists() else {}
        failures = []
        for number, run in enumerate(plan['runs'], 1):
            folder = output / 'runs' / run['id']
            complete = folder / 'complete.json'
            if complete.exists() and statuses.get(run['id'], {}).get('state') == 'complete' and completed_artifacts_valid(folder):
                print(f'[{number}/{len(plan["runs"])} Already complete: {run["id"]}', flush=True)
                continue
            folder.mkdir(parents=True, exist_ok=True)
            write_json(folder / 'run.json', run)
            write_json(folder / 'hardware_start.json', dict(nvidia_smi=command_output(['nvidia-smi']),
                                                           timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            launch = [sys.executable, '-u', '-m', 'experiments.mcb.train', '--run', str(folder / 'run.json'),
                      '--index', str(output / '_data/index.json'), '--data', str(Path(args.data).resolve()), '--output', str(folder)]
            if args.allow_cpu:
                launch += ['--allow-cpu']
            if args.profile == 'smoke':
                launch += ['--sample-limit', '64']
            statuses[run['id']] = dict(state='running', ordinal=number, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write_json(statuses_path, statuses)
            print(f'[{number}/{len(plan["runs"])} {run["group"]}: {run["id"]}', flush=True)
            tick = time.perf_counter()
            with (folder / 'console.log').open('a', encoding='utf-8') as log:
                log.write('\n--- New launch ' + statuses[run['id']]['started_utc'] + ' ---\n')
                log.flush()
                process = subprocess.Popen(launch, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding='utf-8', errors='replace', bufsize=1)
                try:
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        log.write(line)
                        log.flush()
                    code = process.wait()
                except KeyboardInterrupt:
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    statuses[run['id']].update(state='interrupted', wall_s=time.perf_counter()-tick)
                    write_json(statuses_path, statuses)
                    raise
            state = 'complete' if code == 0 and complete.exists() else 'failed'
            write_json(folder / 'hardware_end.json', dict(nvidia_smi=command_output(['nvidia-smi']),
                                                         timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()))
            statuses[run['id']].update(state=state, exit_code=code, wall_s=time.perf_counter()-tick,
                                       ended_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            write_json(statuses_path, statuses)
            collect_raw(output, plan)
            if state == 'failed':
                failures.append(run['id'])
                if args.fail_fast:
                    break
        summary = dict(plan_sha256=plan['plan_sha256'], requested_runs=len(plan['runs']),
                       completed_runs=sum(v['state'] == 'complete' for v in statuses.values()),
                       failed_runs=failures, session_wall_s=time.perf_counter()-started)
        collect_raw(output, plan)
        write_json(output / 'execution_summary.json', summary)
        print('Execution status:', summary, flush=True)
        return 1 if failures else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=('full', 'quick', 'smoke', 'key6'), default='full')
    parser.add_argument('--plan')
    parser.add_argument('--data', default=str(ROOT / 'data/MCB_pointcloud'))
    parser.add_argument('--output')
    parser.add_argument('--cache', default=str(Path.home() / '.cache/mcb-pointnext/packed'))
    parser.add_argument('--no-pack', action='store_true')
    parser.add_argument('--fail-fast', action='store_true')
    parser.add_argument('--allow-cpu', action='store_true', help='Development fixture tests only')
    parser.add_argument('--accept-reconverted-data', action='store_true', help='Accept intentional cross-platform reconversion after identity and full coordinate checks')
    args = parser.parse_args()
    if args.output is None:
        args.output = str(ROOT / 'results' / args.profile)
    try:
        code = run_suite(args)
    except KeyboardInterrupt:
        print('Interrupted; rerun the same BAT to resume the last committed epoch.', flush=True)
        code = 130
    except Exception:
        import traceback
        traceback.print_exc()
        code = 1
    raise SystemExit(code)


if __name__ == '__main__':
    main()
