from pathlib import Path
import numpy as np
import pytest
import torch
from argparse import Namespace
from experiments.mcb.common import DATASETS, MODELS, digest, read_json, sha_file
from experiments.mcb.data import Clouds, corrupt, prepare_data, selected_records
from experiments.mcb.metrics import classification_metrics
from experiments.mcb.models import Classifier
from experiments.mcb.ops import ball_query, grouping_operation
from experiments.mcb.plan import BASE, make_plan
from experiments.mcb.train import train_run, make_loader
from experiments.mcb.suite import run_suite
from experiments.mcb.common import write_json


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / 'clouds'
    rng = np.random.default_rng(100)
    for variant in DATASETS:
        n = int(variant.rsplit('_', 1)[1])
        for label, cls in enumerate(('bearing', 'gear')):
            for split, count in (('train', 3), ('test', 1)):
                folder = root / variant / split / cls
                folder.mkdir(parents=True)
                for i in range(count):
                    model_id = f'{label}_{split}_{i}'
                    x = rng.uniform(-0.4, 0.4, (n, 3)).astype(np.float32)
                    np.save(folder / f'{model_id}.npy', x)
    return root


def test_plan_dedup_and_coverage():
    plan = make_plan('full')
    assert plan['summary']['training_runs'] == 1296
    assert plan['summary']['evaluation_jobs'] == 11898
    assert len({digest(r['config']) for r in plan['runs']}) == len(plan['runs'])
    core = [r for r in plan['runs'] if r['group'] == 'core']
    assert len(core) == 6*9*3
    assert {r['config']['model'] for r in core} == set(MODELS)
    assert all(len(r['evaluations']) == 32 for r in core)


def test_key6_plan_preserves_existing_comparable_runs():
    plan = make_plan('key6')
    assert plan['summary']['training_runs'] == 6
    assert plan['summary']['evaluation_jobs'] == 192
    assert {r['config']['model'] for r in plan['runs']} == set(MODELS)
    assert {r['config']['dataset'] for r in plan['runs']} == {'area_random_2048'}
    quick = {r['id']: r['config'] for r in make_plan('quick')['runs']}
    assert all(quick[r['id']] == r['config'] for r in plan['runs'])


def test_identity_split_and_data_failure(dataset, tmp_path):
    index = prepare_data(dataset, tmp_path / 'inventory', expected_count=0)
    by_split = {s: {r['id'] for r in selected_records(index, s)} for s in ('train', 'val', 'test')}
    assert not by_split['train'] & by_split['val']
    assert not by_split['train'] & by_split['test']
    assert all(sum(r['label'] == label for r in selected_records(index, 'val')) == 1 for label in (0, 1))
    assert prepare_data(dataset, tmp_path / 'inventory', expected_count=0)['data_sha256'] == index['data_sha256']
    path = next((dataset / DATASETS[-1] / 'test/bearing').glob('*.npy'))
    path.unlink()
    with pytest.raises(ValueError, match='Missing'):
        prepare_data(dataset, tmp_path / 'inventory', expected_count=0)


def test_metrics_known_confusion():
    metrics = classification_metrics([0, 0, 1, 1], [[.9, .1], [.2, .8], [.1, .9], [.1, .9]], ['a', 'b'])
    assert metrics['oa'] == .75
    assert metrics['macc'] == .75
    assert metrics['macro_f1'] == pytest.approx((2/3 + .8)/2)
    assert metrics['confusion_matrix'] == [[1, 1], [0, 2]]


def test_corruptions_are_reproducible_finite():
    x = np.random.default_rng(1).normal(size=(256, 3)).astype(np.float32)
    for kind in ('clean', 'jitter', 'rotation', 'so3', 'scale', 'translation', 'dropout', 'outliers', 'occlusion', 'permutation'):
        for severity in (1, 2, 3):
            a = corrupt(x, kind, severity, np.random.default_rng(22))
            b = corrupt(x, kind, severity, np.random.default_rng(22))
            assert np.array_equal(a, b) and a.shape == x.shape and a.dtype == np.float32 and np.isfinite(a).all()
    assert np.array_equal(np.sort(corrupt(x, 'permutation', 1, np.random.default_rng(1)), axis=0), np.sort(x, axis=0))


def test_grouping_values_gradient_and_ball_padding():
    features = torch.tensor([[[2., 4., 6.]]], requires_grad=True)
    idx = torch.tensor([[[0, 0], [1, 2]]], dtype=torch.int32)
    y = grouping_operation(features, idx)
    assert y.tolist() == [[[[2., 2.], [4., 6.]]]]
    y.sum().backward()
    assert features.grad.tolist() == [[[2., 1., 1.]]]
    xyz = torch.tensor([[[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]])
    result = ball_query(.1, 4, xyz, xyz[:, 1:2])
    assert result.tolist() == [[[1, 1, 1, 1]]]


def test_only_reused_training_loader_keeps_workers_alive(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    dataset = torch.utils.data.TensorDataset(torch.ones(4, 3))
    assert make_loader(dataset, 2, 2, training=True).persistent_workers
    validation = make_loader(dataset, 2, 2)
    assert validation.pin_memory and not validation.persistent_workers
    assert not make_loader(dataset, 2, 0, training=True).persistent_workers


@pytest.mark.parametrize('name', MODELS)
def test_complete_architecture_forward_backward(name):
    torch.set_num_threads(2)
    model = Classifier(dict(BASE, model=name))
    y = model(torch.rand(2, 256, 3)*2-1)
    assert y.shape == (2, 32)
    (y.sum()+model.regularizer()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_training_test_resume_records(dataset, tmp_path):
    torch.set_num_threads(2)
    index = prepare_data(dataset, tmp_path / 'inventory', expected_count=0)
    run = make_plan('smoke')['runs'][0]
    run['config'].update(epochs=2, amp=False, workers=0, warmup_epochs=0)
    run['evaluations'] = [run['evaluations'][0], dict(name='jitter_1', dataset=DATASETS[0], corruption='jitter', severity=1)]
    output = tmp_path / 'run'
    train_run(run, index, dataset, output, allow_cpu=True)
    assert read_json(output / 'training_complete.json')['epochs'] == 2
    metric = read_json(output / 'tests/clean.json')
    assert metric['samples'] == 2 and metric['predictions_sha256'] == sha_file(output / 'tests/clean.npz')
    before = sha_file(output / 'best.pt')
    history = (output / 'history.csv').read_bytes()
    train_run(run, index, dataset, output, allow_cpu=True)
    assert before == sha_file(output / 'best.pt')
    assert history == (output / 'history.csv').read_bytes()
    # Repair one interrupted evaluation without changing training/checkpoints.
    (output / 'tests/jitter_1.npz').write_bytes(b'broken')
    train_run(run, index, dataset, output, allow_cpu=True)
    assert before == sha_file(output / 'best.pt')
    assert np.load(output / 'tests/jitter_1.npz')['probabilities'].shape == (2, 2)


def test_actual_interrupted_epoch_resume(dataset, tmp_path, monkeypatch):
    import experiments.mcb.train as training
    torch.set_num_threads(2)
    index = prepare_data(dataset, tmp_path / 'inventory', expected_count=0)
    run = make_plan('smoke')['runs'][0]
    run['config'].update(epochs=2, amp=False, workers=2, warmup_epochs=0)
    run['evaluations'] = [run['evaluations'][0]]
    output = tmp_path / 'interrupted'
    original_save = training.save_checkpoint

    def interrupted_save(path, value):
        original_save(path, value)
        if Path(path).name == 'last.pt' and value['next_epoch'] == 1:
            raise RuntimeError('simulated interruption after epoch one')

    monkeypatch.setattr(training, 'save_checkpoint', interrupted_save)
    with pytest.raises(RuntimeError, match='simulated interruption'):
        train_run(run, index, dataset, output, allow_cpu=True)
    monkeypatch.setattr(training, 'save_checkpoint', original_save)
    train_run(run, index, dataset, output, allow_cpu=True)
    train_run(run, index, dataset, tmp_path / 'continuous', allow_cpu=True)
    a = torch.load(output / 'last.pt', weights_only=False)
    b = torch.load(tmp_path / 'continuous/last.pt', weights_only=False)
    assert a['next_epoch'] == 2
    for key in a['model']:
        torch.testing.assert_close(a['model'][key], b['model'][key], rtol=0, atol=0)


def test_sequential_suite_pack_and_skip(dataset, tmp_path):
    plan = make_plan('smoke')
    run = plan['runs'][4]  # PointNet exercises input/feature transforms in real training.
    run['evaluations'] = [run['evaluations'][0]]
    run['config'].update(amp=False, workers=0, warmup_epochs=0)
    plan['runs'] = [run]
    plan_path = tmp_path / 'plan.json'
    write_json(plan_path, plan)
    args = Namespace(output=str(tmp_path / 'results'), data=str(dataset), cache=str(tmp_path / 'packed'),
                     profile='smoke', plan=str(plan_path), allow_cpu=True, no_pack=False, fail_fast=True)
    assert run_suite(args) == 0
    folder = Path(args.output) / 'runs' / run['id']
    before = sha_file(folder / 'last.pt')
    assert (Path(args.output) / 'results.csv').is_file()
    assert run_suite(args) == 0
    assert sha_file(folder / 'last.pt') == before
    (folder / 'tests/clean.npz').write_bytes(b'broken')
    assert run_suite(args) == 0
    assert sha_file(folder / 'last.pt') == before
    log_size = (folder / 'console.log').stat().st_size
    assert run_suite(args) == 0
    assert (folder / 'console.log').stat().st_size == log_size
