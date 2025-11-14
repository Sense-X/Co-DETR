# Co-DETR 视频人物检测完整教程

本教程介绍如何使用Co-DETR ViT-Large模型对视频中的人物进行检测。

## 📋 目录

1. [环境准备](#环境准备)
2. [模型下载](#模型下载)
3. [使用方法](#使用方法)
4. [Python API调用](#python-api调用)
5. [进阶使用](#进阶使用)

---

## 🔧 环境准备

### 1. 安装依赖

```bash
# 安装MMCV
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

# 安装其他依赖
pip install -r requirements.txt

# 安装Hugging Face CLI (用于下载模型)
pip install huggingface-hub
```

### 2. 验证CUDA可用性

```python
import torch
print(f"CUDA 可用: {torch.cuda.is_available()}")
print(f"CUDA 版本: {torch.version.cuda}")
```

---

## 📥 模型下载

### 方法1: 自动下载（推荐）

使用我们提供的脚本自动从Hugging Face下载：

```bash
python detect_person_in_video.py your_video.mp4 --download
```

### 方法2: 手动下载

从Hugging Face手动下载模型文件：

```bash
# 安装git-lfs
git lfs install

# 克隆模型仓库
git clone https://huggingface.co/zongzhuofan/co-detr-vit-large-coco

# 或使用huggingface-cli下载
huggingface-cli download zongzhuofan/co-detr-vit-large-coco \
    co_dino_5scale_vit_large_coco.pth \
    co_dino_5scale_vit_large_coco.py \
    --local-dir ./checkpoints
```

### 方法3: 使用本地配置文件

如果已经有本地的配置文件和权重：

```bash
python detect_person_in_video.py your_video.mp4 \
    --config projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco.py \
    --checkpoint path/to/checkpoint.pth
```

---

## 🚀 使用方法

### 基础用法

```bash
# 检测视频中的人物
python detect_person_in_video.py input_video.mp4 --output output_video.mp4
```

### 常用参数

```bash
python detect_person_in_video.py input_video.mp4 \
    --output result.mp4 \              # 输出视频路径
    --score-thr 0.5 \                  # 置信度阈值(0-1)
    --device cuda:0 \                  # 使用的GPU设备
    --show                             # 实时显示检测结果
```

### 参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `video` | str | 必需 | 输入视频文件路径 |
| `--config` | str | 自动下载 | 模型配置文件路径 |
| `--checkpoint` | str | 自动下载 | 模型权重文件路径 |
| `--output` | str | `{video}_person_detection.mp4` | 输出视频路径 |
| `--score-thr` | float | 0.5 | 检测置信度阈值 |
| `--device` | str | cuda:0 | 运行设备 |
| `--show` | bool | False | 是否实时显示 |
| `--download` | bool | False | 强制从HF下载 |

---

## 💻 Python API调用

### 示例1: 基础检测

```python
from mmdet.apis import init_detector, inference_detector
import cv2
import mmcv

# 初始化模型
config_file = 'checkpoints/co_dino_5scale_vit_large_coco.py'
checkpoint_file = 'checkpoints/co_dino_5scale_vit_large_coco.pth'
model = init_detector(config_file, checkpoint_file, device='cuda:0')

# 读取视频
video = mmcv.VideoReader('input.mp4')

# 处理每一帧
for frame in video:
    result = inference_detector(model, frame)
    # result是一个列表，result[0]是person类的检测结果
    person_boxes = result[0]  # shape: [N, 5] (x1, y1, x2, y2, score)

    # 可视化
    vis_frame = model.show_result(frame, result, score_thr=0.5)
    cv2.imshow('Detection', vis_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
```

### 示例2: 只检测人物

```python
def filter_person_only(result, score_threshold=0.5):
    """只保留人物检测结果"""
    PERSON_CLASS_ID = 0
    filtered_result = [[] for _ in range(len(result))]

    if len(result[PERSON_CLASS_ID]) > 0:
        person_boxes = result[PERSON_CLASS_ID]
        # 过滤低置信度
        person_boxes = person_boxes[person_boxes[:, -1] > score_threshold]
        filtered_result[PERSON_CLASS_ID] = person_boxes

    return filtered_result

# 使用过滤函数
for frame in video:
    result = inference_detector(model, frame)
    person_result = filter_person_only(result, score_threshold=0.5)

    # 统计人数
    num_persons = len(person_result[0])
    print(f"检测到 {num_persons} 个人")
```

### 示例3: 批量处理

```python
import torch

# 批量推理可以提高速度
batch_frames = []
for i, frame in enumerate(video):
    batch_frames.append(frame)

    # 每32帧处理一次
    if len(batch_frames) == 32 or i == len(video) - 1:
        results = inference_detector(model, batch_frames)

        for frame, result in zip(batch_frames, results):
            # 处理结果
            pass

        batch_frames = []
```

### 示例4: 提取人物信息

```python
import numpy as np

def extract_person_info(result, score_threshold=0.5):
    """提取人物信息"""
    PERSON_CLASS_ID = 0
    person_boxes = result[PERSON_CLASS_ID]

    persons = []
    for box in person_boxes:
        x1, y1, x2, y2, score = box
        if score < score_threshold:
            continue

        person_info = {
            'bbox': [x1, y1, x2, y2],
            'score': score,
            'center': [(x1 + x2) / 2, (y1 + y2) / 2],
            'width': x2 - x1,
            'height': y2 - y1
        }
        persons.append(person_info)

    return persons

# 使用
for frame_idx, frame in enumerate(video):
    result = inference_detector(model, frame)
    persons = extract_person_info(result, score_threshold=0.5)

    print(f"帧 {frame_idx}: 检测到 {len(persons)} 个人")
    for i, person in enumerate(persons):
        print(f"  人物 {i+1}: 置信度={person['score']:.2f}, "
              f"位置=({person['center'][0]:.0f}, {person['center'][1]:.0f})")
```

---

## 🎯 进阶使用

### 1. 调整检测参数

不同场景下推荐的参数设置：

| 场景 | score_thr | 说明 |
|------|-----------|------|
| 高召回率（检测更多人） | 0.3 | 可能有误检 |
| 平衡 | 0.5 | 推荐默认值 |
| 高精度（减少误检） | 0.7 | 可能漏检 |

### 2. 性能优化

```python
# 使用FP16加速推理
from mmcv.runner import wrap_fp16_model
wrap_fp16_model(model)

# 使用TensorRT加速 (需要安装mmcv-full)
# 详见MMDetection文档
```

### 3. 多GPU处理

```bash
# 使用第1个GPU
python detect_person_in_video.py video.mp4 --device cuda:0

# 使用第2个GPU
python detect_person_in_video.py video.mp4 --device cuda:1
```

### 4. CPU推理

```bash
# 如果没有GPU，可以使用CPU（会比较慢）
python detect_person_in_video.py video.mp4 --device cpu
```

### 5. 结果后处理

```python
from mmdet.core import bbox_overlaps

def non_maximum_suppression(boxes, iou_threshold=0.5):
    """非极大值抑制，去除重叠框"""
    if len(boxes) == 0:
        return boxes

    # 按置信度排序
    scores = boxes[:, -1]
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        # 计算IoU
        ious = bbox_overlaps(boxes[i:i+1, :4], boxes[order[1:], :4])[0]

        # 保留IoU小于阈值的框
        inds = np.where(ious <= iou_threshold)[0]
        order = order[inds + 1]

    return boxes[keep]
```

---

## 📊 COCO类别说明

Co-DETR在COCO数据集上训练，支持80个类别。人物检测相关的类别索引：

- **0: person** - 人
- 其他类别（1-79）包括：车辆、动物、物体等

如果需要检测其他类别，修改 `PERSON_CLASS_ID` 即可：

```python
# 检测汽车
CAR_CLASS_ID = 2  # car
car_result = result[CAR_CLASS_ID]

# 检测多个类别
PERSON_CLASS_ID = 0
DOG_CLASS_ID = 16
person_and_dog_result = [result[PERSON_CLASS_ID], result[DOG_CLASS_ID]]
```

完整的COCO类别列表参见：
https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/

---

## ❓ 常见问题

### Q1: 内存不足怎么办？

A: 尝试以下方法：
- 降低视频分辨率
- 减小批处理大小
- 使用FP16推理

### Q2: 检测速度慢怎么办？

A:
- 确保使用GPU: `--device cuda:0`
- 使用更小的模型（如ResNet-50版本）
- 启用FP16或TensorRT加速

### Q3: 误检或漏检怎么办？

A:
- 调整 `score_thr` 参数
- 使用更大的模型（ViT-L效果最好）
- 在特定场景数据上微调模型

### Q4: 如何保存检测结果为JSON？

A:
```python
import json

results_json = []
for frame_idx, frame in enumerate(video):
    result = inference_detector(model, frame)
    persons = extract_person_info(result)
    results_json.append({
        'frame': frame_idx,
        'persons': persons
    })

with open('detection_results.json', 'w') as f:
    json.dump(results_json, f, indent=2)
```

---

## 📚 参考资源

- **论文**: [DETRs with Collaborative Hybrid Assignments Training](https://arxiv.org/pdf/2211.12860.pdf)
- **GitHub**: https://github.com/Sense-X/Co-DETR
- **Hugging Face模型**: https://huggingface.co/zongzhuofan/co-detr-vit-large-coco
- **MMDetection文档**: https://mmdetection.readthedocs.io/

---

## 🎉 完整示例代码

查看 `detect_person_in_video.py` 获取完整实现！

```bash
# 快速开始
python detect_person_in_video.py demo/demo.mp4 --download --show
```
