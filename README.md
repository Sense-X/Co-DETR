# [ICCV 2023] DETRs with Collaborative Hybrid Assignments Training

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/detrs-with-collaborative-hybrid-assignments/object-detection-on-coco-minival)](https://paperswithcode.com/sota/object-detection-on-coco-minival?p=detrs-with-collaborative-hybrid-assignments)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/detrs-with-collaborative-hybrid-assignments/object-detection-on-coco)](https://paperswithcode.com/sota/object-detection-on-coco?p=detrs-with-collaborative-hybrid-assignments)	
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/detrs-with-collaborative-hybrid-assignments/object-detection-on-lvis-v1-0-minival)](https://paperswithcode.com/sota/object-detection-on-lvis-v1-0-minival?p=detrs-with-collaborative-hybrid-assignments)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/detrs-with-collaborative-hybrid-assignments/object-detection-on-lvis-v1-0-val)](https://paperswithcode.com/sota/object-detection-on-lvis-v1-0-val?p=detrs-with-collaborative-hybrid-assignments)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/detrs-with-collaborative-hybrid-assignments/instance-segmentation-on-lvis-v1-0-val)](https://paperswithcode.com/sota/instance-segmentation-on-lvis-v1-0-val?p=detrs-with-collaborative-hybrid-assignments)

This repo is the official implementation of ["DETRs with Collaborative Hybrid Assignments Training"](https://arxiv.org/pdf/2211.12860.pdf) by Zhuofan Zong, Guanglu Song, and Yu Liu.


## News

* [07/14/2023] Co-DETR is accepted to ICCV 2023!
* [07/12/2023] We finetune Co-DETR on LVIS and achieve the best results **without TTA**: **71.2 box AP** and **59.7 mask AP** on LVIS minival, **66.9 box AP** and **56.0 mask AP** on LVIS val. For instance segmentation, we report the performance of the auxiliary mask branch.
* [07/03/2023] Co-DETR with [ViT-L](https://github.com/baaivision/EVA/tree/master/EVA-02) **(304M parameters)** sets a new record of **65.6 AP** on COCO test-dev, surpassing the previous best model [InternImage-G](https://github.com/OpenGVLab/InternImage) **(~3000M parameters)**.
* [07/03/2023] Code for Co-Deformable-DETR is released.
* [11/19/2022] We achieved 64.4 AP on COCO minival and 64.5 AP on COCO test-dev with only ImageNet-1K as pre-training data. Codes will be available soon.
   

## Introduction

![teaser](figures/framework.png)

In this paper, we present a novel collaborative hybrid assignments training scheme, namely Co-DETR, to learn more efficient and effective DETR-based detectors from versatile label assignment manners. 
1. **Encoder optimization**: The proposed training scheme can easily enhance the encoder's learning ability in end-to-end detectors by training multiple parallel auxiliary heads supervised by one-to-many label assignments. 
2. **Decoder optimization**: We conduct extra customized positive queries by extracting the positive coordinates from these auxiliary heads to improve attention learning of the decoder. 
3. **State-of-the-art performance**: Incorporated with the ViT-L backbone (304M parameters),
we achieve 65.6\% AP on MS COCO test-dev, outperforming previous methods with much fewer model sizes.

![teaser](figures/performance.png)

## Model Zoo
### Performance of improved Co-DETR

| Model  | Backbone | Epochs | Queries | Dataset | box AP |
| ------ | -------- | ------ | ------- | ------- | ------ |
| Co-DINO-5scale | R50 | 12 | 900 | COCO | 51.7 |
| Co-DINO-5scale | R50 | 36 | 900 | COCO | 54.0 |
| Co-DINO-5scale-9enc | R50 | 12 | 900 | COCO | 52.1 |
| Co-DINO-5scale-9enc | R50 | 36 | 900 | COCO | 54.7 |
| Co-DINO-5scale | Swin-L | 12 | 900 | COCO | 58.8 |
| Co-DINO-5scale | Swin-L | 36 | 900 | COCO | 60.0 |
| Co-DINO-5scale | Swin-L | 36 | 900 | LVIS | 56.2 |


### Results on Deformable-DETR

| Model  | Backbone | Epochs | Queries | box AP | Ckpt | Log |
| ------ | -------- | ------ | ------- | ------ | ---- | --- |
| Co-Deformable-DETR | R50 | 12 | 300 | 49.5 | [google](https://drive.google.com/drive/folders/1asWoZ3SuM6APTL9D-QUF_YW9mjULNdh9?usp=sharing) | [google](https://drive.google.com/drive/folders/1GktHRm2oAxmOzdK3jPaRqNu4uOQhecgZ?usp=sharing) |
| Co-Deformable-DETR | Swin-T | 12 | 300 | 51.7 | [google](https://drive.google.com/drive/folders/1asWoZ3SuM6APTL9D-QUF_YW9mjULNdh9?usp=sharing) | [google](https://drive.google.com/drive/folders/1GktHRm2oAxmOzdK3jPaRqNu4uOQhecgZ?usp=sharing) |
| Co-Deformable-DETR | Swin-T | 36 | 300 | 54.1 | [google](https://drive.google.com/drive/folders/1asWoZ3SuM6APTL9D-QUF_YW9mjULNdh9?usp=sharing) | [google](https://drive.google.com/drive/folders/1GktHRm2oAxmOzdK3jPaRqNu4uOQhecgZ?usp=sharing) |
| Co-Deformable-DETR | Swin-S | 12 | 300 | 53.4 | [google](https://drive.google.com/drive/folders/1asWoZ3SuM6APTL9D-QUF_YW9mjULNdh9?usp=sharing) | [google](https://drive.google.com/drive/folders/1GktHRm2oAxmOzdK3jPaRqNu4uOQhecgZ?usp=sharing) |
| Co-Deformable-DETR | Swin-S | 36 | 300 | 55.3 | [google](https://drive.google.com/drive/folders/1asWoZ3SuM6APTL9D-QUF_YW9mjULNdh9?usp=sharing) | [google](https://drive.google.com/drive/folders/1GktHRm2oAxmOzdK3jPaRqNu4uOQhecgZ?usp=sharing) |
| Co-Deformable-DETR | Swin-B | 12 | 300 | 55.5 | [google](https://drive.google.com/drive/folders/1asWoZ3SuM6APTL9D-QUF_YW9mjULNdh9?usp=sharing) | [google](https://drive.google.com/drive/folders/1GktHRm2oAxmOzdK3jPaRqNu4uOQhecgZ?usp=sharing) |
| Co-Deformable-DETR | Swin-B | 36 | 300 | 57.5 | [google](https://drive.google.com/drive/folders/1asWoZ3SuM6APTL9D-QUF_YW9mjULNdh9?usp=sharing) | [google](https://drive.google.com/drive/folders/1GktHRm2oAxmOzdK3jPaRqNu4uOQhecgZ?usp=sharing) |
| Co-Deformable-DETR | Swin-L | 12 | 300 | 56.9 | [google](https://drive.google.com/drive/folders/1asWoZ3SuM6APTL9D-QUF_YW9mjULNdh9?usp=sharing) | [google](https://drive.google.com/drive/folders/1GktHRm2oAxmOzdK3jPaRqNu4uOQhecgZ?usp=sharing) |
| Co-Deformable-DETR | Swin-L | 36 | 900 | 58.5 | [google](https://drive.google.com/drive/folders/1asWoZ3SuM6APTL9D-QUF_YW9mjULNdh9?usp=sharing) | [google](https://drive.google.com/drive/folders/1GktHRm2oAxmOzdK3jPaRqNu4uOQhecgZ?usp=sharing) |

## Running

### Install
We implement Co-DETR using [MMDetection V2.25.3](https://github.com/open-mmlab/mmdetection/releases/tag/v2.25.3) and [MMCV V1.5.0](https://github.com/open-mmlab/mmcv/releases/tag/v1.5.0).
The source code of MMdetection has been included in this repo and you only need to build MMCV following [official instructions](https://github.com/open-mmlab/mmcv/tree/v1.5.0#installation).
We test our models under ```python=3.7.11,pytorch=1.11.0,cuda=11.3```. Other versions may not be compatible. 

### Data
The COCO dataset should be organized as:
```
data/
  ├── train2017/
  ├── val2017/
  └── annotations/
  	├── instances_train2017.json
  	└── instances_val2017.json
```

### Training
Train Co-Deformable-DETR + ResNet-50 with 8 GPUs:
```shell
sh tools/dist_train.sh projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py 8 path_to_exp
```
Train using slurm:
```shell
sh tools/slurm_train.sh partition job_name projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py path_to_exp
```

### Testing
Test Co-Deformable-DETR + ResNet-50 with 8 GPUs, and evaluate:
```shell
sh tools/dist_test.sh  projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py  path_to_checkpoint 8 --eval bbox
```
Test using slurm:
```shell
sh tools/slurm_test.sh partition job_name projects/configs/co_deformable_detr/co_deformable_detr_r50_1x_coco.py path_to_checkpoint --eval bbox
```

## Cite Co-DETR

If you find this repository useful, please use the following BibTeX entry for citation.

```latex
@misc{codetr2022,
      title={DETRs with Collaborative Hybrid Assignments Training},
      author={Zhuofan Zong and Guanglu Song and Yu Liu},
      year={2022},
      eprint={2211.12860},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License

This project is released under the MIT license. Please see the [LICENSE](LICENSE) file for more information.
