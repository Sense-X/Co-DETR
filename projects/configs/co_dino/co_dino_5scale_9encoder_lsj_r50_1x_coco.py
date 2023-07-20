_base_ = [
    'co_dino_5scale_lsj_r50_1x_coco.py'
]
# model settings

model = dict(
    query_head=dict(
        transformer=dict(
            encoder=dict(
                type='DetrTransformerEncoder',
                num_layers=9,
                # number of layers that use checkpoint
                with_cp=9))))