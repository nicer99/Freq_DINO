_base_ = 'grounding_dino_swin-b_pretrain_obj365_goldg_v3det.py'

import os

# Grounding DINO few-shot baseline for DSA defect detection.
# This baseline freezes the text encoder and the full Swin-B visual backbone,
# then fine-tunes the detection adaptation modules.

custom_imports = dict(imports=['my_custom_module3'], allow_failed_imports=False)

# Relative defaults are resolved from the repository root by the launcher.
data_root = os.path.join(os.path.expanduser(
    os.environ.get('DSA_DATA_ROOT', 'data/dsa')), '')
weights_dir = os.path.expanduser(
    os.environ.get('FREQ_DINO_WEIGHTS_DIR', 'weights/checkpoint'))
lang_model_name = os.path.expanduser(
    os.environ.get('FREQ_DINO_BERT', 'bert-base-uncased'))
load_from = os.path.join(
    weights_dir,
    'grounding_dino_swin-b_pretrain_obj365_goldg_v3de-f83eef00.pth')

class_names = (
    'Missing_Hole', 'Connected_hole', 'Block', 'Particle', 'Dislocation')
metainfo = dict(classes=class_names)

model = dict(
    type='GroundingDINOLoRA',
    lora_r=0,
    lora_alpha=16,
    lora_dropout=0.0,
    lora_target_modules=[],
    lora_exclude_modules=['language_model'],
    freeze_non_lora=True,
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
    ],
    use_freq_prior=False,
    freq_channels=256,
    freq_num_levels=4,
    freq_center_radius=30,
    freq_point_radius=8,
    freq_outlier_percentile=0.5,
    freq_smooth_sigma=20,
    # Number of independent Fourier peak pairs. Each selected peak is
    # suppressed together with its conjugate-symmetric counterpart.
    freq_num_peaks=3,
    freq_border_ratio=0.03,
    language_model=dict(
        type='BertModel',
        name=lang_model_name,
        pad_to_max=False,
        use_sub_sentence_represent=True,
        special_tokens_list=['[CLS]', '[SEP]', '.', '?'],
        add_pooling_layer=False),
    backbone=dict(init_cfg=None, frozen_stages=-1),
    bbox_head=dict(num_classes=256, contrastive_cfg=dict(max_text_len=256)))

_multiscale_resize = [
    dict(
        type='RandomChoiceResize',
        scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                (736, 1333), (768, 1333), (800, 1333)],
        keep_ratio=True)
]

_multiscale_resize_with_crop = [
    dict(
        type='RandomChoiceResize',
        scales=[(400, 4200), (500, 4200), (600, 4200)],
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=(384, 600),
        allow_negative_crop=True),
    dict(
        type='RandomChoiceResize',
        scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333),
                (608, 1333), (640, 1333), (672, 1333), (704, 1333),
                (736, 1333), (768, 1333), (800, 1333)],
        keep_ratio=True)
]

train_pipeline_od = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[_multiscale_resize, _multiscale_resize_with_crop]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='RandomSamplingNegPos',
        tokenizer_name=lang_model_name,
        num_sample_negative=85,
        label_map_file=data_root + 'label_map.json',
        max_tokens=256),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode')),
]

train_pipeline_vg = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='LoadTokensPositive'),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[_multiscale_resize, _multiscale_resize_with_crop]),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode')),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=(800, 1333), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='FixTextForVal',
        text='Missing_Hole . Connected_hole . Block . Particle . '
             'Dislocation .'),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities')),
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),
    dataset=dict(
        _delete_=True,
        type='ConcatDataset',
        datasets=[
            dict(
                type='ODVGDataset',
                data_root=data_root,
                ann_file='few_shot_dataset/train_10shot_seed1.jsonl',
                label_map_file='label_map.json',
                data_prefix=dict(img='images/'),
                filter_cfg=dict(filter_empty_gt=False),
                pipeline=train_pipeline_od,
                return_classes=True),
            dict(
                type='ODVGDataset',
                data_root=data_root,
                ann_file='few_shot_dataset/train_10shot_seed1.jsonl',
                label_map_file=None,
                data_prefix=dict(img='images/'),
                filter_cfg=dict(filter_empty_gt=False),
                pipeline=train_pipeline_vg,
                return_classes=True),
        ]))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='few_shot_dataset/val_10shot_seed1.json',
        data_prefix=dict(img='images/'),
        test_mode=True,
        pipeline=test_pipeline,
        return_classes=True,
        metainfo=metainfo,
        backend_args=None))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'few_shot_dataset/val_10shot_seed1.json',
    metric='bbox',
    classwise=True)
test_evaluator = val_evaluator

max_epochs = 120

optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=5e-5, weight_decay=1e-4),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(custom_keys={
        'language_model': dict(lr_mult=0.0),
        'backbone.patch_embed': dict(lr_mult=0.0),
        'backbone.stages.0': dict(lr_mult=0.0),
        'backbone.stages.1': dict(lr_mult=0.0),
        'backbone.stages.2': dict(lr_mult=0.0),
        'backbone.stages.3': dict(lr_mult=0.0),
        'backbone.norm': dict(lr_mult=0.0),
    }))

param_scheduler = [
    dict(type='LinearLR', start_factor=0.001, by_epoch=False, begin=0, end=80),
    dict(
        type='CosineAnnealingLR',
        by_epoch=True,
        begin=0,
        end=max_epochs,
        eta_min=1e-6)
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=max_epochs,
                 val_interval=10)
val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

default_hooks = dict(
    timer=dict(type='IterTimerHook'),
    logger=dict(type='LoggerHook', interval=10),
    param_scheduler=dict(type='ParamSchedulerHook'),
    checkpoint=dict(
        type='CheckpointHook',
        interval=10,
        max_keep_ckpts=2,
        # Historical checkpoint saving uses AP. Paper tables select the
        # AP50 maximum from validation logs (AP breaks ties); see docs.
        save_best='coco/bbox_mAP',
        rule='greater'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook'),
)

custom_hooks = [dict(type='CheckFreezeHook')]

env_cfg = dict(
    cudnn_benchmark=False,
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    dist_cfg=dict(backend='nccl'))
vis_backends = [dict(type='LocalVisBackend')]
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')
log_processor = dict(type='LogProcessor', window_size=10, by_epoch=True)
log_level = 'INFO'
randomness = dict(seed=42, deterministic=False)
