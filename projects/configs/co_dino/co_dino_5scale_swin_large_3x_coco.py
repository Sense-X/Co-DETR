_base_ = [
    'co_dino_5scale_swin_large_1x_coco.py'
]
# model settings
model = dict(
    backbone=dict(drop_path_rate=0.6))

lr_config = dict(policy='step', step=[30])
runner = dict(type='EpochBasedRunner', max_epochs=36)
