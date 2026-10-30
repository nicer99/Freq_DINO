_base_ = './dsa_fewshot_gray_branch.py'

# Queue-friendly version for repeated few-shot ablations. The model is identical
# to dsa_fewshot_gray_branch.py, but visualization/stat hooks are disabled to
# avoid concurrent writes when multiple GPUs run different sampling seeds.
custom_hooks = [
    dict(type='CheckFreezeHook'),
]
