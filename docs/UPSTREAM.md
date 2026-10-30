# Upstream source and local changes

The bundled framework comes from
[OpenMMLab MMDetection](https://github.com/open-mmlab/mmdetection), revision
`cfd5d3a985b0249de009b67d04f37263e11cdf3d` (`v3.3.0-5-gcfd5d3a9`).
The package includes the Python framework and the configuration dependencies
needed by the Freq-DINO training entries. It does not include the upstream Git
history, model files, example images, or the authors' experiment artifacts.

The framework has two functional local changes retained from the research code:

| File | Change |
|---|---|
| `mmdet/datasets/transforms/text_transformers.py` | Replaces obsolete `np.long` with `int` for NumPy compatibility. |
| `mmdet/evaluation/metrics/coco_metric.py` | Skips predictions whose label index is greater than or equal to the number of configured dataset categories, with a warning for bbox predictions. |

The second change affects validation when out-of-range labels occur. Such
predictions are omitted from COCO export; they are not counted as valid class
predictions. This behavior is retained to match the local training workflow.
Investigate the class/prompt mapping if the warning appears instead of treating
the validation metrics as unaffected. Garbled comments were removed without
changing that logic.

Temporary debug printing in `ODVGDataset` was removed. The detector's trailing
newline-only change was also normalized to upstream. These two cleanups do not
change training behavior. The method implementation itself is copied from the
current research source without algorithm changes.

Project-specific training configurations replace machine-local paths with
environment variables or command-line arguments. The training launcher sets
the requested shot and support-set seed in both OD and VG dataset views and in
the matching validation annotation file.

Upstream license text and existing copyright notices are retained. Original
Freq-DINO components have no author-selected license at this time; see
[LICENSE.md](../LICENSE.md).
