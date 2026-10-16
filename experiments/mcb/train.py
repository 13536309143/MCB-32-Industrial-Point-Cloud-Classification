import argparse
import csv
import io
import math
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, RandomSampler
from .common import atomic_bytes, digest, read_json, seed_for, sha_file, write_json
from .data import Clouds, selected_records
from .metrics import classification_metrics
from .models import Classifier


def seed_everything(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    # Official grouping backward uses atomicAdd. warn_only retains these kernels;
    # bitwise deterministic GPU training is not promised.
    torch.use_deterministic_algorithms(deterministic, warn_only=True)


def sync(device):
    if device.type == 'cuda':
        torch.cuda.synchronize(device)


def autocast_options(config, device):
    enabled = config['amp'] and device.type == 'cuda'
    requested = config.get('amp_dtype', 'auto')
    if requested == 'auto':
        requested = 'bfloat16' if device.type == 'cuda' and torch.cuda.is_bf16_supported() else 'float16'
    dtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16}[requested]
    return dict(device_type=device.type, enabled=enabled, dtype=dtype)


def gradient_scaler(config, device):
    options = autocast_options(config, device)
    # BF16 has FP32-like exponent range; loss scaling is needed only for FP16.
    return torch.amp.GradScaler('cuda', enabled=options['enabled'] and options['dtype'] == torch.float16)


def save_checkpoint(path, value):
    buffer = io.BytesIO()
    torch.save(value, buffer)
    atomic_bytes(path, buffer.getvalue())


def limit_stratified(records, limit):
    if not limit:
        return records
    groups = {}
    for record in records:
        groups.setdefault(record['label'], []).append(record)
    selected = []
    for position in range(max(map(len, groups.values()))):
        for label in sorted(groups):
            if position < len(groups[label]):
                selected.append(groups[label][position])
                if len(selected) >= limit:
                    return selected
    return selected


def loss_vector(logits, labels, config, weights=None):
    loss = F.cross_entropy(logits.float(), labels, reduction='none', label_smoothing=config['label_smoothing'])
    if config['loss'] == 'weighted_ce':
        loss = loss * weights[labels]
    elif config['loss'] == 'focal':
        pt = torch.softmax(logits.float(), 1).gather(1, labels[:, None]).squeeze(1)
        loss = loss * (1-pt)**2
    return loss


def make_loader(dataset, batch, workers, shuffle=False, generator=None, training=False):
    # Keep sample ordering independent of DataLoader's worker base-seed consumption.
    # Persistent workers consume that seed differently after a fresh-process resume.
    sampler = RandomSampler(dataset, generator=generator) if shuffle else None
    worker_generator = torch.Generator().manual_seed(42)
    return DataLoader(dataset, batch_size=batch, sampler=sampler, num_workers=workers,
                      drop_last=training, pin_memory=torch.cuda.is_available(),
                      # Only the single reused training loader keeps workers alive.
                      # Repeated validation/test loaders with persistent workers +
                      # pinning retain atexit worker handles in PyTorch (FD leak).
                      persistent_workers=training and workers > 0, generator=worker_generator)


def choose_batch(config, device, classes):
    seed_everything(config['seed'], config['deterministic'])
    n = int(config['dataset'].rsplit('_', 1)[1])
    candidates = [config['micro_batch']] if config['micro_batch'] else [min(32, config['effective_batch']), 16, 8, 4, 2]
    candidates = sorted(set(b for b in candidates if 2 <= b <= config['effective_batch']
                            and config['effective_batch'] % b == 0), reverse=True)
    attempts = []
    for batch in candidates:
        try:
            allocated = probe_batch(config, device, classes, batch, n)
            if device.type == 'cuda' and not config['micro_batch'] and batch > 2:
                total = torch.cuda.get_device_properties(device).total_memory
                if allocated > 0.8*total:
                    attempts.append(dict(batch=batch, status='insufficient_20pct_headroom', peak_allocated_bytes=allocated))
                    torch.cuda.empty_cache()
                    continue
            attempts.append(dict(batch=batch, status='ok', peak_allocated_bytes=allocated))
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            return batch, attempts
        except torch.OutOfMemoryError:
            attempts.append(dict(batch=batch, status='cuda_oom'))
            if device.type == 'cuda':
                torch.cuda.empty_cache()
    raise RuntimeError(f'Cannot train with physical batch >= 2: {attempts}. No model/point-count reduction was made.')


def probe_batch(config, device, classes, batch, n):
    # Separate frame so all tensors/optimizer states are released after an OOM.
    model = Classifier(config, classes).to(device).train()
    amp = config['amp'] and device.type == 'cuda'
    scaler = gradient_scaler(config, device)
    optimizer_class = dict(adamw=torch.optim.AdamW, adam=torch.optim.Adam, sgd=torch.optim.SGD)[config['optimizer']]
    options = dict(lr=config['lr'], weight_decay=config['weight_decay'])
    if config['optimizer'] == 'sgd':
        options.update(momentum=0.9, nesterov=True)
    optimizer = optimizer_class(model.parameters(), **options)
    xyz = torch.randn(batch, n, 3, device=device).clamp(-1, 1)
    target = torch.arange(batch, device=device) % classes
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(2):
        with torch.autocast(**autocast_options(config, device)):
            loss = loss_vector(model(xyz), target, config, torch.ones(classes, device=device)).mean() + model.regularizer()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    sync(device)
    return torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0


@torch.inference_mode()
def evaluate(model, dataset, classes, config, weights, device, batch, workers):
    model.eval()
    loader = make_loader(dataset, batch, workers)
    probs, labels, ids = [], [], []
    loss_sum, compute_s = 0.0, 0.0
    sync(device)
    started = time.perf_counter()
    for xyz, target, model_ids in loader:
        xyz, target = xyz.to(device, non_blocking=True), target.to(device, non_blocking=True)
        sync(device)
        tick = time.perf_counter()
        with torch.autocast(**autocast_options(config, device)):
            logits = model(xyz)
            losses = loss_vector(logits, target, config, weights)
        sync(device)
        compute_s += time.perf_counter()-tick
        loss_sum += losses.double().sum().item()
        probs.append(logits.float().softmax(1).cpu().numpy())
        labels.extend(target.cpu().tolist())
        ids.extend(model_ids)
    sync(device)
    elapsed = time.perf_counter()-started
    probs = np.concatenate(probs)
    labels = np.asarray(labels)
    result = classification_metrics(labels, probs, classes)
    result.update(loss=loss_sum/len(labels), elapsed_s=elapsed, forward_loss_compute_s=compute_s,
                  samples_per_second=len(labels)/elapsed, batch_size=batch)
    return result, dict(model_id=np.asarray(ids), y_true=labels.astype(np.int64),
                        y_pred=probs.argmax(1).astype(np.int64), probabilities=probs.astype(np.float32))


def benchmark(model, xyz, config, device):
    model.eval()
    output = {}
    for batch in (1, 8):
        sample = xyz[:1].expand(batch, -1, -1).contiguous().to(device)
        try:
            timings = []
            with torch.inference_mode(), torch.autocast(**autocast_options(config, device)):
                for _ in range(10):
                    model(sample)
                sync(device)
                if device.type == 'cuda':
                    torch.cuda.reset_peak_memory_stats(device)
                for _ in range(30):
                    tick = time.perf_counter()
                    model(sample)
                    sync(device)
                    timings.append((time.perf_counter()-tick)*1000)
            output[f'batch_{batch}'] = dict(warmup=10, repeats=30, milliseconds=timings,
                                           p50_ms=float(np.median(timings)), p95_ms=float(np.percentile(timings, 95)),
                                           throughput_samples_s=batch*1000/float(np.mean(timings)),
                                           peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0)
        except torch.OutOfMemoryError:
            output[f'batch_{batch}'] = dict(status='cuda_oom')
            torch.cuda.empty_cache()
    return output


def learning_rate(config, epoch):
    if epoch < config['warmup_epochs']:
        return config['lr']*(epoch+1)/config['warmup_epochs']
    if config['scheduler'] == 'step':
        return max(config['min_lr'], config['lr']*(0.7**(epoch//20)))
    phase = (epoch-config['warmup_epochs'])/max(1, config['epochs']-config['warmup_epochs']-1)
    return config['min_lr']+(config['lr']-config['min_lr'])*0.5*(1+math.cos(math.pi*phase))


def train_run(run, index, data_root, output, allow_cpu=False, sample_limit=0):
    cfg = run['config']
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda' and not allow_cpu:
        raise RuntimeError('CUDA is required for actual experiments; CPU fallback is disabled.')
    records = selected_records(index, 'train', cfg['train_fraction'])
    val_records = selected_records(index, 'val')
    if sample_limit:
        records, val_records = limit_stratified(records, sample_limit), limit_stratified(val_records, sample_limit)
    classes = index['classes']
    weights = np.bincount([r['label'] for r in records], minlength=len(classes))
    weights = np.sqrt(1 / np.maximum(weights, 1))
    weights /= weights.mean()
    weights = torch.tensor(weights, device=device, dtype=torch.float32)
    binding = dict(config_sha256=digest(cfg), data_sha256=index['data_sha256'],
                   split_sha256=index['split_sha256'], sample_limit=sample_limit,
                   device_type=device.type)
    old_binding = output / 'binding.json'
    if old_binding.exists() and read_json(old_binding) != binding:
        raise RuntimeError('Existing results belong to different data/config/split/device. Use a new result directory.')
    write_json(old_binding, binding)
    write_json(output / 'config.json', run)
    last = output / 'last.pt'
    completed = output / 'training_complete.json'
    resume = torch.load(last, map_location='cpu', weights_only=False) if last.exists() else None
    if resume:
        micro = resume['micro_batch']
        attempts = resume['batch_probe']
    else:
        micro, attempts = choose_batch(cfg, device, len(classes))
    seed_everything(cfg['seed'], cfg['deterministic'])
    model = Classifier(cfg, len(classes)).to(device)
    write_json(output / 'model_spec.json', model.spec)
    optimizer_class = dict(adamw=torch.optim.AdamW, adam=torch.optim.Adam, sgd=torch.optim.SGD)[cfg['optimizer']]
    options = dict(lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    if cfg['optimizer'] == 'sgd':
        options.update(momentum=0.9, nesterov=True)
    optimizer = optimizer_class(model.parameters(), **options)
    scaler = gradient_scaler(cfg, device)
    train_set = Clouds(data_root, cfg['dataset'], records, cfg['seed'], cfg['augmentation'])
    val_set = Clouds(data_root, cfg['dataset'], val_records, cfg['seed'])
    generator = torch.Generator()
    train_loader = make_loader(train_set, micro, cfg['workers'], shuffle=True, generator=generator, training=True)
    start_epoch, best_key, elapsed_total, compute_total = 0, (-1., -1., -float('inf')), 0., 0.
    if resume:
        model.load_state_dict(resume['model'])
        optimizer.load_state_dict(resume['optimizer'])
        scaler.load_state_dict(resume['scaler'])
        start_epoch, best_key = resume['next_epoch'], tuple(resume['best_key'])
        elapsed_total, compute_total = resume['elapsed_total_s'], resume['compute_total_s']
    epochs_folder = output / 'epochs'
    epochs_folder.mkdir(exist_ok=True)
    # Remove only our own stale epoch record, if interrupted between record and checkpoint.
    for p in epochs_folder.glob('*.json'):
        if int(p.stem) >= start_epoch:
            p.unlink()
    if not completed.exists():
        for epoch in range(start_epoch, cfg['epochs']):
            seed_everything(seed_for(cfg['seed'], 'epoch', epoch), cfg['deterministic'])
            generator.manual_seed(seed_for(cfg['seed'], 'loader', epoch))
            train_set.epoch = epoch
            model.train()
            for group in optimizer.param_groups:
                group['lr'] = learning_rate(cfg, epoch)
            optimizer.zero_grad(set_to_none=True)
            pending, total_loss, total_samples = 0, 0., 0
            all_probs, all_labels, events = [], [], []
            sync(device)
            if device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(device)
            started = time.perf_counter()
            for step, (xyz, target, _) in enumerate(train_loader):
                xyz, target = xyz.to(device, non_blocking=True), target.to(device, non_blocking=True)
                if device.type == 'cuda':
                    before, after = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    before.record()
                else:
                    before = time.perf_counter()
                with torch.autocast(**autocast_options(cfg, device)):
                    logits = model(xyz)
                    per_sample = loss_vector(logits, target, cfg, weights)
                    loss = per_sample.sum() + model.regularizer()*len(target)
                scaler.scale(loss).backward()
                pending += len(target)
                if pending >= cfg['effective_batch'] or step == len(train_loader)-1:
                    scaler.unscale_(optimizer)
                    for parameter in model.parameters():
                        if parameter.grad is not None:
                            parameter.grad.div_(pending)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    pending = 0
                if device.type == 'cuda':
                    after.record()
                    events.append((before, after))
                else:
                    events.append(time.perf_counter()-before)
                total_loss += loss.detach().item()
                total_samples += len(target)
                all_probs.append(logits.detach().float().softmax(1).cpu().numpy())
                all_labels.extend(target.cpu().tolist())
            sync(device)
            train_elapsed = time.perf_counter()-started
            compute_elapsed = sum(a.elapsed_time(b)/1000 for a, b in events) if device.type == 'cuda' else sum(events)
            elapsed_total += train_elapsed
            compute_total += compute_elapsed
            train_metrics = classification_metrics(all_labels, np.concatenate(all_probs), classes)
            train_metrics.update(loss=total_loss/total_samples, elapsed_s=train_elapsed, compute_s=compute_elapsed,
                                 dropped_samples=len(records)-total_samples)
            val, _ = evaluate(model, val_set, classes, cfg, weights, device, micro, cfg['workers'])
            key = (val['macro_f1'], val['oa'], -val['loss'])
            improved = key > best_key
            if improved:
                best_key = key
                save_checkpoint(output / 'best.pt', dict(model=model.state_dict(), epoch=epoch, val_metrics=val, binding=binding))
            row = dict(epoch=epoch, lr=optimizer.param_groups[0]['lr'], train=train_metrics, val=val,
                       best_checkpoint_updated=improved, peak_allocated_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0)
            write_json(epochs_folder / f'{epoch:04d}.json', row)
            save_checkpoint(last, dict(model=model.state_dict(), optimizer=optimizer.state_dict(), scaler=scaler.state_dict(),
                                      next_epoch=epoch+1, best_key=best_key, micro_batch=micro, batch_probe=attempts,
                                      elapsed_total_s=elapsed_total, compute_total_s=compute_total, binding=binding))
            print(f"Epoch {epoch+1}/{cfg['epochs']} train_loss={train_metrics['loss']:.5f} val_OA={val['oa']:.5f} val_F1={val['macro_f1']:.5f} {train_elapsed:.1f}s", flush=True)
        write_json(completed, dict(epochs=cfg['epochs'], training_wall_s=elapsed_total, training_compute_s=compute_total,
                                  physical_batch=micro, effective_batch=cfg['effective_batch'], batch_probe=attempts,
                                  parameters=sum(p.numel() for p in model.parameters()), training_samples=len(records),
                                  validation_samples=len(val_records), fraction_ids_sha256=digest([r['id'] for r in records]),
                                  amp=cfg['amp'] and device.type == 'cuda',
                                  actual_precision=str(autocast_options(cfg, device)['dtype']) if cfg['amp'] and device.type == 'cuda' else 'torch.float32',
                                  class_loss_weights=weights.cpu().tolist()))
    # Generate a flat, raw epoch table; no averaging/ranking/analysis.
    flat_rows = []
    for path in sorted(epochs_folder.glob('*.json')):
        row = read_json(path)
        flat = dict(epoch=row['epoch'], lr=row['lr'], peak_allocated_bytes=row['peak_allocated_bytes'])
        for split in ('train', 'val'):
            flat.update({split+'_'+key: val for key, val in row[split].items() if isinstance(val, (int, float))})
        flat_rows.append(flat)
    if flat_rows:
        with (output / 'history.csv').open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(flat_rows[0]))
            writer.writeheader()
            writer.writerows(flat_rows)
    best = torch.load(output / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(best['model'])
    checkpoint_hash = sha_file(output / 'best.pt')
    write_json(output / 'best_checkpoint.json', dict(epoch=best['epoch'], val_metrics=best['val_metrics'], sha256=checkpoint_hash))
    benchmark_path = output / 'benchmark.json'
    if not benchmark_path.exists():
        example = torch.from_numpy(val_set[0][0]).unsqueeze(0)
        write_json(benchmark_path, benchmark(model, example, cfg, device))
    test_records = selected_records(index, 'test')
    if sample_limit:
        test_records = limit_stratified(test_records, sample_limit)
    tests = output / 'tests'
    tests.mkdir(exist_ok=True)
    for case in run['evaluations']:
        name = case['name']
        metric_path, predictions_path = tests / f'{name}.json', tests / f'{name}.npz'
        if metric_path.exists() and predictions_path.exists():
            if read_json(metric_path).get('predictions_sha256') == sha_file(predictions_path):
                continue
        dataset = Clouds(data_root, case['dataset'], test_records, cfg['seed'],
                         corruption=case['corruption'], severity=case['severity'])
        eval_batch = micro
        while True:
            try:
                metrics, predictions = evaluate(model, dataset, classes, cfg, weights, device, eval_batch, cfg['workers'])
                break
            except torch.OutOfMemoryError:
                if eval_batch == 1:
                    raise
                eval_batch = max(1, eval_batch//2)
                torch.cuda.empty_cache()
        buffer = io.BytesIO()
        np.savez_compressed(buffer, **predictions)
        atomic_bytes(predictions_path, buffer.getvalue())
        metrics.update(case=case, checkpoint_epoch=best['epoch'], checkpoint_sha256=checkpoint_hash,
                       predictions_sha256=sha_file(predictions_path), train_dataset=cfg['dataset'],
                       cross_domain_definition='same held-out test identities, different sampling/point-count representation')
        write_json(metric_path, metrics)
        print(f"Test {name}: loss={metrics['loss']:.5f} OA={metrics['oa']:.5f} mAcc={metrics['macc']:.5f} F1={metrics['macro_f1']:.5f}", flush=True)
    artifacts = {}
    for path in sorted(output.rglob('*')):
        if path.is_file() and path.suffix in ('.json', '.npz', '.pt', '.csv') and path.name not in (
                'complete.json', 'run.json', 'hardware_start.json', 'hardware_end.json'):
            stat = path.stat()
            artifacts[path.relative_to(output).as_posix()] = dict(size=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=sha_file(path))
    write_json(output / 'complete.json', dict(run_id=run['id'], evaluations=len(run['evaluations']), binding=binding,
                                              artifacts=artifacts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    parser.add_argument('--index', required=True)
    parser.add_argument('--data', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--allow-cpu', action='store_true')
    parser.add_argument('--sample-limit', type=int, default=0)
    args = parser.parse_args()
    train_run(read_json(args.run), read_json(args.index), args.data, args.output, args.allow_cpu, args.sample_limit)


if __name__ == '__main__':
    main()
