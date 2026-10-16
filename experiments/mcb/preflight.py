"""Fail before long training if any complete model or CUDA backward is broken."""
import argparse
import json
import torch
from .common import MODELS
from .models import Classifier
from .ops import extension, grouping_operation, ball_query, furthest_point_sample
from .plan import BASE
from .train import autocast_options


def check(allow_cpu=False):
    if not torch.cuda.is_available() and not allow_cpu:
        raise RuntimeError('CUDA is required; check Windows NVIDIA driver, WSL2, and cu128 PyTorch.')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        extension()
        # Check exact grouping values and gradients against a pure PyTorch gather.
        features = torch.randn(2, 8, 256, device=device, requires_grad=True)
        idx = torch.randint(256, (2, 16, 8), device=device, dtype=torch.int32)
        actual = grouping_operation(features, idx)
        expected = features.gather(2, idx.long().reshape(2, 1, -1).expand(2, 8, -1)).reshape(2, 8, 16, 8)
        torch.testing.assert_close(actual, expected)
        a = torch.autograd.grad(actual.sum(), features, retain_graph=True)[0]
        b = torch.autograd.grad(expected.sum(), features)[0]
        torch.testing.assert_close(a, b)
        xyz = torch.rand(2, 256, 3, device=device)
        fps = furthest_point_sample(xyz, 64)
        centers = xyz.gather(1, fps.long()[..., None].expand(-1, -1, 3))
        neighbors = ball_query(0.2, 16, xyz, centers)
        assert neighbors.min() >= 0 and neighbors.max() < 256
    results = []
    for name in MODELS:
        config = dict(BASE, model=name)
        model = Classifier(config).to(device).train()
        xyz = torch.rand(2, 2048 if device.type == 'cuda' else 256, 3, device=device)*2-1
        for amp in (False, True) if device.type == 'cuda' else (False,):
            model.zero_grad(set_to_none=True)
            with torch.autocast(**autocast_options(dict(config, amp=amp), device)):
                logits = model(xyz)
                loss = torch.nn.functional.cross_entropy(logits.float(), torch.tensor([0, 1], device=device)) + model.regularizer()
            assert logits.shape == (2, 32) and torch.isfinite(logits).all()
            loss.backward()
            assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        info = dict(model=name, parameters=sum(p.numel() for p in model.parameters()), device=str(device))
        print('Forward/backward OK:', info, flush=True)
        results.append(info)
        del model, xyz, logits, loss
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--allow-cpu', action='store_true')
    args = parser.parse_args()
    print(json.dumps(check(args.allow_cpu), indent=2))
