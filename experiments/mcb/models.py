"""Load only classification dependencies from the complete, untouched upstream tree.

The official package __init__ imports unrelated segmentation/reconstruction backbones,
requiring additional Linux extensions. Selective imports avoid those dependencies.
"""
import ast
import copy
import importlib.util
import logging
import sys
import types
from typing import List
import torch
from torch import nn
import yaml
from easydict import EasyDict
from .common import ROOT
from . import ops

_loaded = False


def package(name, path):
    mod = types.ModuleType(name)
    mod.__path__ = [str(path)]
    sys.modules[name] = mod
    return mod


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def upstream():
    global _loaded
    if _loaded:
        return sys.modules['openpoints.models.build'], sys.modules['openpoints.models.classification.cls_base']
    folder = ROOT / 'PointNeXt' / 'openpoints'
    if not (folder / 'models/backbone/pointnext.py').is_file():
        raise RuntimeError('Copy the complete PointNeXt/openpoints submodule with this project.')
    package('openpoints', folder)
    utilities = package('openpoints.utils', folder / 'utils')
    utilities.registry = load('openpoints.utils.registry', folder / 'utils/registry.py')
    package('openpoints.models', folder / 'models')
    build = load('openpoints.models.build', folder / 'models/build.py')
    cpp = package('openpoints.cpp', folder / 'cpp')
    batch = package('openpoints.cpp.pointnet2_batch', folder / 'cpp/pointnet2_batch')
    # Imports need a namespace; every executed classification operation is replaced below.
    cpp.pointnet2_cuda = batch.pointnet2_cuda = types.SimpleNamespace()
    layers = package('openpoints.models.layers', folder / 'models/layers')
    for name in ('activation', 'norm', 'conv'):
        module = load('openpoints.models.layers.' + name, folder / f'models/layers/{name}.py')
        for attr in ('create_act', 'create_norm', 'create_convblock1d', 'create_convblock2d', 'create_linearblock'):
            if hasattr(module, attr):
                setattr(layers, attr, getattr(module, attr))
    group = load('openpoints.models.layers.group', folder / 'models/layers/group.py')
    group.ball_query = ops.ball_query
    group.grouping_operation = ops.grouping_operation
    group.gather_operation = ops.gather_operation
    for attr in ('create_grouper', 'get_aggregation_feautres'):
        setattr(layers, attr, getattr(group, attr))
    layers.furthest_point_sample = ops.furthest_point_sample
    layers.random_sample = ops.random_sample
    up = load('openpoints.models.layers.upsampling', folder / 'models/layers/upsampling.py')
    layers.three_interpolation = up.three_interpolation  # decoder-only; never executed in this suite
    aggregation = load('openpoints.models.layers.local_aggregation', folder / 'models/layers/local_aggregation.py')
    layers.CHANNEL_MAP = aggregation.CHANNEL_MAP
    layers.LocalAggregation = aggregation.LocalAggregation
    package('openpoints.models.backbone', folder / 'models/backbone')
    for name in ('pointnext', 'pointnet', 'pointnetv2'):
        load('openpoints.models.backbone.' + name, folder / f'models/backbone/{name}.py')
    # Execute the complete, unchanged official ClsHead class; BaseCls/DistillCls imports
    # checkpoint/loss modules for unrelated tasks. AST extraction preserves ClsHead code.
    path = folder / 'models/classification/cls_base.py'
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ClsHead')
    package('openpoints.models.classification', path.parent)
    head = types.ModuleType('openpoints.models.classification.cls_base')
    head.__file__ = str(path)
    head.__dict__.update(nn=nn, List=List, logging=logging, MODELS=build.MODELS,
                         create_linearblock=layers.create_linearblock)
    sys.modules[head.__name__] = head
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), 'exec'), head.__dict__)
    _loaded = True
    return build, head


def model_spec(config):
    name = config['model']
    configs = ROOT / 'PointNeXt/cfgs'
    if name == 'pointnext_s':
        path = configs / 'modelnet40ply2048/pointnext-s.yaml'
    elif name.startswith('pointnext_'):
        path = configs / f"s3dis/pointnext-{name.split('_')[-1]}.yaml"
    elif name == 'pointnet2':
        path = configs / 'modelnet40ply2048/pointnet++_original.yaml'
    elif name == 'pointnet':
        path = configs / 'scanobjectnn/pointnet.yaml'
    else:
        raise ValueError(name)
    args = copy.deepcopy(yaml.safe_load(path.read_text(encoding='utf-8'))['model']['encoder_args'])
    args['in_channels'] = 3
    if name.startswith('pointnext_'):
        args.update(radius=config['radius'], nsample=config['nsample'])
    return dict(encoder=args, upstream_config=str(path.relative_to(ROOT)),
                head=dict(NAME='ClsHead', num_classes=32, mlps=[512, 256],
                          norm_args={'norm': 'bn1d'}, dropout=config['dropout']),
                pool=config['pool'])


class Classifier(nn.Module):
    def __init__(self, config, num_classes=32):
        super().__init__()
        build, _ = upstream()
        self.spec = model_spec(config)
        self.kind = config['model']
        self.pool = config['pool']
        if self.pool not in ('max', 'max_avg') or (self.pool == 'max_avg' and not self.kind.startswith('pointnext_')):
            raise ValueError('max_avg is a PointNeXt head ablation; the baseline encoders already perform global pooling.')
        self.encoder = build.build_model_from_cfg(EasyDict(copy.deepcopy(self.spec['encoder'])))
        self.transform = None
        if self.kind == 'pointnet' and self.encoder.fstn is not None:
            self.encoder.fstn.register_forward_hook(self._transform_hook)
        head = copy.deepcopy(self.spec['head'])
        head.update(num_classes=num_classes, in_channels=self.encoder.out_channels*(2 if self.pool == 'max_avg' else 1))
        self.spec['head'] = copy.deepcopy(head)
        self.head = build.build_model_from_cfg(EasyDict(head))

    def _transform_hook(self, module, inputs, output):
        self.transform = output

    def forward(self, xyz):
        xyz = xyz.float().contiguous()
        if self.kind.startswith('pointnext_'):
            _, features = self.encoder.forward_seg_feat(xyz)
            x = features[-1]
            maximum = x.max(-1).values
            x = torch.cat((maximum, x.mean(-1)), dim=1) if self.pool == 'max_avg' else maximum
        else:
            x = self.encoder.forward_cls_feat(xyz)
        return self.head(x)

    def regularizer(self):
        if self.kind != 'pointnet' or self.transform is None:
            return 0.0
        transform = self.transform.float()
        eye = torch.eye(transform.shape[1], device=transform.device)[None]
        return 0.001 * torch.linalg.matrix_norm(transform @ transform.transpose(1, 2) - eye).mean()
