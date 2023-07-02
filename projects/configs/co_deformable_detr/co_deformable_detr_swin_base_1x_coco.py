_base_ = [
    'co_deformable_detr_r50_1x_coco.py'
]
pretrained = 'models/swin_base_patch4_window12_384_22k.pth'
# model settings
model = dict(
    backbone=dict(
        _delete_=True,
        type='SwinTransformerV1',
        embed_dim=128,
        depths=[2, 2, 18, 2],
        num_heads=[4, 8, 16, 32],
        out_indices=(1, 2, 3),
        window_size=12,
        ape=False,
        drop_path_rate=0.2,
        patch_norm=True,
        use_checkpoint=False,
        pretrained=pretrained),
    neck=dict(in_channels=[128*2, 128*4, 128*8]))

# optimizer
optimizer = dict(weight_decay=0.05)
