_base_ = [
    'co_dino_5scale_r50_1x_coco.py'
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
        out_indices=(0, 1, 2, 3),
        window_size=12,
        ape=False,
        drop_path_rate=0.3,
        patch_norm=True,
        use_checkpoint=False,
        pretrained=pretrained),
    neck=dict(in_channels=[192, 192*2, 192*4, 192*8]),
    query_head=dict(
        transformer=dict(
            encoder=dict(
                # number of layers that use checkpoint
                with_cp=6))))

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=1)