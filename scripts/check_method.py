"""CPU smoke check of the real prior, encoder and fusion on synthetic tensors."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

    import torch
    from my_custom_module3 import (
        FreqPriorModule, GatedFusionModule, ResidualFusionModule)

    torch.set_num_threads(1)
    torch.manual_seed(42)
    images = torch.randn(1, 3, 128, 128)
    sizes = [(16, 16), (8, 8), (4, 4), (2, 2)]
    prior = FreqPriorModule()
    heatmap = prior._compute_heatmap_batch(images)
    assert heatmap.shape == (1, 1, 128, 128)
    assert torch.isfinite(heatmap).all()
    assert heatmap.min() >= 0 and heatmap.max() <= 1.000001
    assert not heatmap.requires_grad
    assert torch.equal(heatmap, prior._compute_heatmap_batch(images))
    blank = prior._compute_heatmap_batch(torch.zeros_like(images))
    assert torch.isfinite(blank).all() and torch.count_nonzero(blank) == 0

    encoded = prior(images, sizes)
    visual = [torch.randn_like(f) for f in encoded]
    fusion = GatedFusionModule()
    fused = fusion(visual, encoded)
    initial_gate = torch.sigmoid(torch.tensor(-2.0))
    for index, (value, source, frequency) in enumerate(zip(fused, visual, encoded)):
        expected = source + initial_gate * fusion.freq_norms[index](frequency)
        assert value.shape[-2:] == sizes[index]
        assert torch.isfinite(value).all()
        torch.testing.assert_close(value, expected)

    loss = sum(value.square().mean() for value in fused)
    loss.backward()
    grad = prior.encoder[0].weight.grad
    assert grad is not None and torch.isfinite(grad).all()
    assert torch.count_nonzero(grad) > 0
    for gate in fusion.gates:
        grad = gate[3].weight.grad
        assert grad is not None and torch.isfinite(grad).all()
        assert torch.count_nonzero(grad) > 0

    with torch.no_grad():
        gray = FreqPriorModule(prior_type='gray')(images, sizes)
        residual = ResidualFusionModule()(visual, gray)
        assert all(torch.isfinite(value).all() for value in residual)

    print('PASS: deterministic FFT prior, blank input, four-scale fusion, '
          'initial gate, gradients, and grayscale/residual control.')
    print('CPU synthetic tensors only; no dataset, checkpoint, or model download.')


if __name__ == '__main__':
    main()
