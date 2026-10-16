"""Classify prepared float32 XYZ NPY files using a trained run's best checkpoint."""
import argparse
import csv
from pathlib import Path
import numpy as np
import torch
from .common import read_json
from .models import Classifier
from .train import autocast_options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run-dir', required=True)
    parser.add_argument('--input', required=True, help='One NPY or a directory of prepared NPY clouds')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('Prediction requires CUDA in the prepared Windows/WSL environment.')
    folder = Path(args.run_dir)
    cfg = read_json(folder / 'config.json')['config']
    index = read_json(folder.parents[1] / '_data/index.json')
    model = Classifier(cfg, len(index['classes'])).cuda().eval()
    checkpoint = torch.load(folder / 'best.pt', map_location='cuda', weights_only=False)
    model.load_state_dict(checkpoint['model'])
    source = Path(args.input)
    paths = [source] if source.is_file() else sorted(source.rglob('*.npy'))
    if not paths:
        raise ValueError('No NPY files found')
    rows = []
    expected_n = int(cfg['dataset'].rsplit('_', 1)[1])
    with torch.inference_mode():
        for path in paths:
            x = np.load(path, allow_pickle=False)
            if x.dtype != np.float32 or x.shape != (expected_n, 3) or not np.isfinite(x).all():
                raise ValueError(f'Expected finite float32 ({expected_n}, 3): {path}')
            xyz = torch.from_numpy(np.ascontiguousarray(x)).cuda()[None]
            with torch.autocast(**autocast_options(cfg, torch.device('cuda'))):
                probabilities = model(xyz).float().softmax(1)[0].cpu().numpy()
            predicted = int(probabilities.argmax())
            row = dict(file=str(path), predicted_idx=predicted, predicted_class=index['classes'][predicted],
                       confidence=float(probabilities[predicted]))
            row.update({f'p_{c}': float(probabilities[i]) for i, c in enumerate(index['classes'])})
            rows.append(row)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Predictions saved:', output)


if __name__ == '__main__':
    main()
