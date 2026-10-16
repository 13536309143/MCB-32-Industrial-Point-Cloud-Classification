"""Modern allocation/AMP wrappers around the unmodified official CUDA kernels.

CPU references are ONLY for --allow-cpu tests, never a training fallback.
"""
import importlib
import torch
from torch.autograd import Function

_extension = None


def extension():
    global _extension
    if _extension is None:
        _extension = importlib.import_module('mcb_pointnet2_cuda')
    return _extension


@torch.no_grad()
def furthest_point_sample(xyz, npoint):
    xyz = xyz.float().contiguous()
    b, n, _ = xyz.shape
    if not 0 < npoint <= n:
        raise ValueError(f'Invalid FPS request {npoint} / {n}')
    if xyz.is_cuda:
        output = torch.empty((b, npoint), device=xyz.device, dtype=torch.int32)
        temp = torch.full((b, n), 1e10, device=xyz.device, dtype=torch.float32)
        extension().furthest_point_sampling_wrapper(b, n, npoint, xyz, temp, output)
        return output
    output = torch.empty((b, npoint), device=xyz.device, dtype=torch.int64)
    dist = torch.full((b, n), 1e10, device=xyz.device)
    farthest = torch.zeros(b, dtype=torch.int64, device=xyz.device)
    batch = torch.arange(b, device=xyz.device)
    for i in range(npoint):
        output[:, i] = farthest
        d = ((xyz - xyz[batch, farthest, None])**2).sum(-1)
        dist = torch.minimum(dist, d)
        farthest = dist.argmax(-1)
    return output


@torch.no_grad()
def ball_query(radius, nsample, xyz, new_xyz):
    xyz, new_xyz = xyz.float().contiguous(), new_xyz.float().contiguous()
    b, n, _ = xyz.shape
    m = new_xyz.shape[1]
    if xyz.is_cuda:
        idx = torch.zeros((b, m, nsample), device=xyz.device, dtype=torch.int32)
        extension().ball_query_wrapper(b, n, m, float(radius), nsample, new_xyz, xyz, idx)
        return idx
    # Official ball query takes the first K indices, not nearest K; pads with first hit.
    indices = torch.arange(n, device=xyz.device).view(1, 1, n).expand(b, m, -1)
    valid = ((new_xyz[:, :, None] - xyz[:, None])**2).sum(-1) < radius**2
    marked = indices.masked_fill(~valid, n)
    chosen = marked.topk(min(nsample, n), dim=-1, largest=False, sorted=True).values
    first = chosen[..., :1].clamp(max=n-1)
    first = torch.where(first == n-1, torch.where(valid.any(-1, keepdim=True), first, 0), first)
    chosen = torch.where(chosen == n, first, chosen)
    if chosen.shape[-1] < nsample:
        chosen = torch.cat((chosen, first.expand(b, m, nsample-chosen.shape[-1])), -1)
    return chosen.int().contiguous()


class CUDAGrouping(Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type='cuda', cast_inputs=torch.float32)
    def forward(ctx, features, idx):
        features, idx = features.contiguous(), idx.int().contiguous()
        b, c, n = features.shape
        m, k = idx.shape[1:]
        output = torch.empty((b, c, m, k), device=features.device, dtype=torch.float32)
        extension().group_points_wrapper(b, c, n, m, k, features, idx, output)
        ctx.save_for_backward(idx)
        ctx.n = n
        return output

    @staticmethod
    @torch.amp.custom_bwd(device_type='cuda')
    def backward(ctx, grad):
        idx, = ctx.saved_tensors
        b, c, m, k = grad.shape
        output = torch.zeros((b, c, ctx.n), device=grad.device, dtype=torch.float32)
        extension().group_points_grad_wrapper(b, c, ctx.n, m, k, grad.float().contiguous(), idx, output)
        return output, None


def grouping_operation(features, idx):
    if features.is_cuda:
        return CUDAGrouping.apply(features.float(), idx)
    b, c, _ = features.shape
    return features.gather(2, idx.long().reshape(b, 1, -1).expand(b, c, -1)).reshape(b, c, *idx.shape[1:])


def gather_operation(features, idx):
    return features.gather(2, idx.long().unsqueeze(1).expand(-1, features.shape[1], -1))


def random_sample(xyz, npoint):
    # Preserve the official internal random sampler behavior (with replacement).
    return torch.randint(xyz.shape[1], (xyz.shape[0], npoint), device=xyz.device)
