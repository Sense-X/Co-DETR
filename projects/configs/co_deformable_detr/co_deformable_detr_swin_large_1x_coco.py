_base_ = [
    'co_deformable_detr_r50_1x_coco.py'
]
pretrained = 'models/swin_large_patch4_window12_384_22k.pth'
# model settings
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformerV1',
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        out_indices=(1, 2, 3),
        window_size=12,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False,
        pretrained=pretrained),
    neck=dict(in_channels=[192*2, 192*4, 192*8]))

# optimizer
optimizer = dict(weight_decay=0.05)
