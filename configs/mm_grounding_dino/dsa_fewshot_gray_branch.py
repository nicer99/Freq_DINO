_base_ = './dsa_fewshot_freq_prior.py'

# Ablation: replace the Fourier residual heatmap with a normalized grayscale
# image prior, while keeping the same lightweight encoder and gated fusion.
# This controls for extra trainable parameters in the prior branch.

model = dict(
    freq_prior_type='gray',
    freq_fusion_type='gated')

custom_hooks = [
    dict(type='CheckFreezeHook'),
    dict(
        type='FreqPriorVisualizeHook',
        save_dir='work_dirs/ablation_vis/gray_branch',
        vis_interval=10,
        num_images=8,
        feat_level=0),
    dict(
        type='FreqFusionStatsHook',
        save_dir='work_dirs/ablation_stats/gray_branch',
        stats_interval=10),
]
