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
        drop_path_rate=0.6,
        patch_norm=True,
        use_checkpoint=False,
        pretrained=pretrained),
    neck=dict(in_channels=[192*2, 192*4, 192*8]),
    query_head=dict(num_query=900),
    test_cfg=[
        dict(max_per_img=300),
        dict(
            rpn=dict(
                nms_pre=1000,
                max_per_img=1000,
                nms=dict(type='nms', iou_threshold=0.7),
                min_bbox_size=0),
            rcnn=dict(
                score_thr=0.0,
                nms=dict(type='nms', iou_threshold=0.5),
                max_per_img=100)),
        dict(
            nms_pre=1000,
            min_bbox_size=0,
            score_thr=0.0,
            nms=dict(type='nms', iou_threshold=0.6),
            max_per_img=100),
        # soft-nms is also supported for rcnn testing
        # e.g., nms=dict(type='soft_nms', iou_threshold=0.5, min_score=0.05)
    ])

# optimizer
optimizer = dict(weight_decay=0.05)
lr_config = dict(policy='step', step=[30])
runner = dict(type='EpochBasedRunner', max_epochs=36)