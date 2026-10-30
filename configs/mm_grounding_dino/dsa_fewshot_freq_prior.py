_base_ = './dsa_fewshot_base.py'

# Proposed method: Grounding DINO baseline plus the frequency-prior branch and
# multi-scale gated residual fusion.

model = dict(
    use_freq_prior=True,
    trainable_patterns=[
        'encoder.layers',
        'encoder.text_layers',
        'encoder.fusion_layers',
        'neck',
        'decoder',
        'bbox_head',
        'query_embedding',
        'text_feat_map',
        'memory_trans',
        'dn_query_generator',
        'level_embed',
        'freq_prior',
        'gated_fusion',
    ])

custom_hooks = [
    dict(type='CheckFreezeHook'),
    dict(
        type='FreqPriorVisualizeHook',
        save_dir='work_dirs/freq_prior_vis',
        vis_interval=10,
        num_images=8,
        feat_level=0),
    dict(
        type='FreqFusionStatsHook',
        save_dir='work_dirs/freq_prior_stats',
        stats_interval=10),
]
