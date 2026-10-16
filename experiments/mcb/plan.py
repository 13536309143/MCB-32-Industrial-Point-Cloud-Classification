"""An explicit, deterministic plan. No selection using test results."""
import argparse
from collections import Counter
from itertools import product
from .common import DATASETS, MODELS, VERSION, digest, write_json

BASE = dict(epochs=300, lr=0.001, weight_decay=0.05, optimizer='adamw',
            scheduler='cosine', warmup_epochs=10, min_lr=1e-6,
            effective_batch=32, micro_batch=0, label_smoothing=0.1,
            loss='ce', dropout=0.5, augmentation='standard',
            radius=0.15, nsample=32, pool='max', grad_clip=1.0,
            train_fraction=1.0, amp=True, amp_dtype='auto', workers=4, val_fraction=0.1,
            split_seed=20261013, deterministic=False)


def eval_cases(config, extended=False):
    cases = [dict(name='clean', dataset=config['dataset'], corruption='clean', severity=0)]
    if extended:
        for target in DATASETS:
            if target != config['dataset']:
                cases.append(dict(name='cross_' + target, dataset=target, corruption='clean', severity=0))
        for corruption in ('jitter', 'rotation', 'scale', 'translation', 'dropout', 'outliers', 'occlusion'):
            for severity in (1, 2, 3):
                cases.append(dict(name=f'{corruption}_{severity}', dataset=config['dataset'],
                                  corruption=corruption, severity=severity))
        cases.append(dict(name='permutation', dataset=config['dataset'], corruption='permutation', severity=1))
        cases.append(dict(name='so3', dataset=config['dataset'], corruption='so3', severity=1))
    return cases


def make_plan(profile='full'):
    runs = []
    seen = set()

    def add(group, model, dataset, seed=42, extended=False, **updates):
        cfg = dict(BASE, model=model, dataset=dataset, seed=seed, **updates)
        key = digest(cfg)
        if key in seen:
            return
        seen.add(key)
        run_id = f'{model}__{dataset}__s{seed}__{key[:12]}'
        runs.append(dict(id=run_id, group=group, config=cfg, evaluations=eval_cases(cfg, extended)))

    if profile == 'key6':
        for model in MODELS:
            add('key6', model, 'area_random_2048', extended=True)
    elif profile == 'smoke':
        for model in MODELS:
            add('smoke', model, 'area_random_2048', epochs=1, effective_batch=4,
                micro_batch=2, workers=0, extended=True)
    else:
        seeds = (42,) if profile == 'quick' else (42, 123, 2026)
        for model, dataset, seed in product(MODELS, DATASETS, seeds):
            add('core', model, dataset, seed, extended=True)
        if profile == 'full':
            # One factor at a time; the base configuration is already in core.
            changes = [dict(lr=x) for x in (0.0003, 0.003)]
            changes += [dict(weight_decay=x) for x in (0.0, 0.001, 0.01)]
            changes += [dict(effective_batch=x) for x in (16, 64)]
            changes += [dict(radius=x) for x in (0.075, 0.3)]
            changes += [dict(nsample=x) for x in (16, 64)]
            changes += [dict(label_smoothing=x) for x in (0.0, 0.2)]
            changes += [dict(dropout=0.2), dict(pool='max_avg'), dict(augmentation='none'),
                        dict(augmentation='so3'), dict(augmentation='strong'),
                        dict(loss='weighted_ce'), dict(loss='focal'),
                        dict(optimizer='sgd', lr=0.01, weight_decay=0.0001),
                        dict(scheduler='step'), dict(amp=False), dict(epochs=600)]
            for model, dataset, change in product(MODELS[:4], DATASETS, changes):
                add('pointnext_sweep', model, dataset, **change,
                    extended=change.get('augmentation') in ('so3', 'strong'))
            # Interaction grid: base value + alternate value for each factor.
            for model, dataset, lr, wd, batch in product(MODELS[:4], DATASETS,
                    (0.001, 0.003), (0.01, 0.05), (32, 64)):
                add('interaction_grid', model, dataset, lr=lr, weight_decay=wd, effective_batch=batch)
            # Three methods at fixed 4096 points, same retained model IDs for every method/model.
            for model, dataset, fraction, seed in product(MODELS[:4],
                    ('area_random_4096', 'fps_4096', 'poisson_disk_4096'), (0.1, 0.25, 0.5), seeds):
                add('data_efficiency', model, dataset, seed, train_fraction=fraction, extended=True)
            # Brief conventional-recipe baseline, separate from fair common-recipe core.
            for model, dataset in product(MODELS[4:], DATASETS):
                add('baseline_classic', model, dataset, optimizer='adam', weight_decay=0.0001,
                    scheduler='step', label_smoothing=0.0, warmup_epochs=0, augmentation='classic')
    summary = dict(training_runs=len(runs), evaluation_jobs=sum(len(r['evaluations']) for r in runs),
                   groups=dict(Counter(r['group'] for r in runs)),
                   total_training_epochs=sum(r['config']['epochs'] for r in runs))
    plan = dict(version=VERSION, profile=profile, summary=summary, runs=runs)
    plan['plan_sha256'] = digest(plan)
    return plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--profile', choices=('full', 'quick', 'smoke', 'key6'), default='full')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    plan = make_plan(args.profile)
    write_json(args.output, plan)
    print(plan['summary'])


if __name__ == '__main__':
    main()
