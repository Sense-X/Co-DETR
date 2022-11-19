# DETRs with Collaborative Hybrid Assignments Training

This repo is the official implementation of "DETRs with Collaborative Hybrid Assignments Training" by Zhuofan Zong, Guanglu Song, and Yu Liu.


## News

* [11/19/2022] We achieved 64.4 AP on COCO minival and 64.5 AP on COCO test-dev with only ImageNet-1K as pre-training data. Codes will be available soon.
   

## Introduction

In this paper,
we present a novel collaborative hybrid assignments training scheme, namely Co-DETR, to learn more efficient and effective DETR-based detectors from versatile label assignment manners. This new training scheme can easily enhance the encoder's learning ability in end-to-end detectors by training the multiple parallel auxiliary heads supervised by one-to-many label assignments. In addition, we conduct extra customized positive queries by extracting the positive coordinates from these auxiliary heads to improve the training efficiency of positive samples in decoder. Extensive experiments on MS COCO dataset demonstrate the efficiency and effectiveness of our Co-DETR. Surprisingly, incorporated with the large-scale backbone MixMIM-g with 1-Billion parameters, we achieve the 64.5\% AP on MS COCO test dev, achieving superior performance with much fewer extra data sizes.

![teaser](figures/framework.png)

## Results on COCO with ResNet-50

| Model  | Backbone | Epochs | Queries | K | AP |
| ------ | -------- | ------ | ------- | - | -- |
| Deformable-DETR | R50 | 12 | 300 | 0 | 37.1 |
| Deformable-DETR | R50 | 36 | 300 | 0 | 43.3 |
| Co-Deformable-DETR | R50 | 12 | 300 | 1 | 42.3 |
| Co-Deformable-DETR | R50 | 36 | 300 | 1 | 46.8 |
| Co-Deformable-DETR | R50 | 12 | 300 | 2 | 42.9 |
| Co-Deformable-DETR | R50 | 36 | 300 | 2 | 46.5 |


For Deformable-DETR++, we follow the settings in [H-DETR](https://github.com/HDETR/H-Deformable-DETR).

| Model  | Backbone | Epochs | Queries | K | AP |
| ------ | -------- | ------ | ------- | - | -- |
| Deformable-DETR++ | R50 | 12 | 300 | 0 | 47.1 |
| Co-Deformable-DETR | R50 | 12 | 300 | 1 | 48.7 |
| Co-Deformable-DETR | R50 | 12 | 300 | 2 | 49.5 |
| H-Deformable-DETR | R50 | 12 | 300 | 0 | 48.4 |
| Co-H-Deformable-DETR | R50 | 12 | 300 | 1 | 49.2 |
| Co-H-Deformable-DETR | R50 | 12 | 300 | 2 | 49.7 |

## Results on COCO with Swin Transformer

| Model  | Backbone | Epochs | Queries | AP |
| ------ | -------- | ------ | ------- | -- |
| Deformable-DETR++ | Swin-T | 12 | 300 | 49.8 |
| Deformable-DETR++ | Swin-B | 12 | 300 | 54.0 |
| Deformable-DETR++ | Swin-L | 12 | 300 | 55.2 |

| Model  | Backbone | Epochs | Queries | AP (K=1) | AP (K=2) |
| ------ | -------- | ------ | ------- | -------- | -------- |
| Co-Deformable-DETR | Swin-T | 12 | 300 | 51.6 | 51.7 |
| Co-Deformable-DETR | Swin-T | 36 | 300 | 53.9 | 54.1 |
| Co-Deformable-DETR | Swin-S | 12 | 300 | 53.4 | 53.4 |
| Co-Deformable-DETR | Swin-S | 36 | 300 | 55.0 | 55.3 |
| Co-Deformable-DETR | Swin-B | 12 | 300 | 55.2 | 55.5 |
| Co-Deformable-DETR | Swin-B | 36 | 300 | 57.0 | 57.5 |
| Co-Deformable-DETR | Swin-L | 12 | 300 | 56.4 | 56.9 |
| Co-Deformable-DETR | Swin-L | 36 | 900 | 58.1 | 58.3 |
| Co-Deformable-DETR (top 300) | Swin-L | 36 | 900 | 58.3 | 58.5 |



## License

This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.
