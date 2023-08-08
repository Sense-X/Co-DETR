_base_ = [
    'co_dino_5scale_lsj_swin_large_1x_coco.py'
]
# model settings
model = dict(
    backbone=dict(drop_path_rate=0.5))

lr_config = dict(policy='step', step=[20])
runner = dict(type='EpochBasedRunner', max_epochs=24)
