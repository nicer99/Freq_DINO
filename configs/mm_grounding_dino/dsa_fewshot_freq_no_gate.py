_base_ = './dsa_fewshot_freq_prior.py'

# Ablation: use the Fourier frequency prior branch, but remove the learned
# spatial gate. The normalized frequency feature is injected by residual
# addition: F_out = F_v + GN(F_f).

model = dict(
    freq_fusion_type='residual')

custom_hooks = [
    dict(type='CheckFreezeHook'),
    dict(
        type='FreqPriorVisualizeHook',
        save_dir='work_dirs/ablation_vis/freq_no_gate',
        vis_interval=10,
        num_images=8,
        feat_level=0),
    dict(
        type='FreqFusionStatsHook',
        save_dir='work_dirs/ablation_stats/freq_no_gate',
        stats_interval=10),
]
