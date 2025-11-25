# Co-DETR 视频人物检测 - 快速开始指南 🚀

用Co-DETR ViT-Large模型检测视频中的人物，只需3步！

---

## 📦 准备工作

### 环境要求
- Python >= 3.7
- PyTorch >= 1.11.0
- CUDA 11.3 (GPU推荐，CPU也可用)

### 安装依赖
```bash
# 1. 安装MMCV
pip install mmcv-full==1.5.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11.0/index.html

# 2. 安装项目依赖
pip install -r requirements.txt

# 3. 安装Hugging Face CLI
pip install huggingface-hub
```

---

## 🎯 三步快速开始

### 第一步: 下载模型

```bash
# 使用Hugging Face CLI下载模型权重
huggingface-cli download zongzhuofan/co-detr-vit-large-coco \
    pytorch_model.pth \
    --local-dir ./checkpoints
```

**下载后的文件结构：**
```
checkpoints/
└── pytorch_model.pth  (~2.2GB)

# 配置文件使用项目自带的：
projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco.py
```

### 第二步: 准备视频

将你的视频放到项目目录，或者使用demo视频：
```bash
# 使用项目自带的demo视频
ls demo/demo.mp4
```

### 第三步: 运行检测

#### 方法A: 使用命令行工具（推荐）

```bash
# 基础用法
python detect_person_in_video.py 你的视频.mp4 --download

# 完整参数
python detect_person_in_video.py input.mp4 \
    --output result.mp4 \        # 输出路径
    --score-thr 0.5 \            # 置信度阈值
    --device cuda:0 \            # 使用GPU
    --show                       # 实时显示
```

#### 方法B: 使用快速入门脚本

```bash
# 1. 编辑配置
nano quick_start_person_detection.py
# 修改 VIDEO_PATH = '你的视频路径'

# 2. 运行
python quick_start_person_detection.py
```

---

## 💡 常用示例

### 示例1: 检测并保存视频
```bash
python detect_person_in_video.py input.mp4 --output output.mp4 --score-thr 0.5
```

### 示例2: 实时显示检测过程
```bash
python detect_person_in_video.py input.mp4 --show
```

### 示例3: 高精度检测（减少误检）
```bash
python detect_person_in_video.py input.mp4 --score-thr 0.7
```

### 示例4: 高召回率检测（检测更多人）
```bash
python detect_person_in_video.py input.mp4 --score-thr 0.3
```

### 示例5: CPU运行（无GPU）
```bash
python detect_person_in_video.py input.mp4 --device cpu
```

---

## 📊 参数说明

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `video` | 输入视频路径 | 必需 | `input.mp4` |
| `--output` | 输出视频路径 | 自动生成 | `result.mp4` |
| `--score-thr` | 置信度阈值 | 0.5 | `0.7` (严格) / `0.3` (宽松) |
| `--device` | 运行设备 | cuda:0 | `cuda:1` / `cpu` |
| `--show` | 实时显示 | False | 添加此参数启用 |
| `--download` | 自动下载模型 | False | 添加此参数启用 |

---

## 🐍 Python代码示例

### 最简单的例子

```python
from mmdet.apis import init_detector, inference_detector
import cv2
import mmcv

# 1. 加载模型
model = init_detector(
    'projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco.py',  # 配置文件
    'checkpoints/pytorch_model.pth',  # 权重文件
    device='cuda:0'
)

# 2. 读取视频
video = mmcv.VideoReader('input.mp4')

# 3. 逐帧检测
for frame in video:
    result = inference_detector(model, frame)

    # result[0] 是person类的检测结果
    # 格式: [[x1, y1, x2, y2, score], ...]
    person_boxes = result[0]

    print(f"检测到 {len(person_boxes)} 个人")
```

### 完整功能示例

```python
from mmdet.apis import init_detector, inference_detector
import cv2
import mmcv

# 配置
VIDEO_PATH = 'input.mp4'
OUTPUT_PATH = 'output.mp4'
SCORE_THRESHOLD = 0.5

# 初始化模型
model = init_detector(
    'projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco.py',  # 配置文件
    'checkpoints/pytorch_model.pth',  # 权重文件
    device='cuda:0'
)

# 打开视频
video_reader = mmcv.VideoReader(VIDEO_PATH)
video_writer = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*'mp4v'),
    video_reader.fps,
    (video_reader.width, video_reader.height)
)

# 处理每一帧
for frame in mmcv.track_iter_progress(video_reader):
    # 推理
    result = inference_detector(model, frame)

    # 只保留person类
    person_result = [[] for _ in range(len(result))]
    person_boxes = result[0]
    person_boxes = person_boxes[person_boxes[:, -1] > SCORE_THRESHOLD]
    person_result[0] = person_boxes

    # 可视化
    vis_frame = model.show_result(
        frame, person_result,
        score_thr=SCORE_THRESHOLD,
        bbox_color=(0, 255, 0)
    )

    # 保存
    video_writer.write(vis_frame)

video_writer.release()
print(f"完成！结果保存到: {OUTPUT_PATH}")
```

---

## ❓ 常见问题

### Q: 如何只检测人，不检测其他物体？

A: 脚本已经默认只检测人物（person类，索引为0）。如果需要检测其他类别：

```python
# COCO数据集类别索引
# 0: person, 1: bicycle, 2: car, 16: dog, 17: cat ...
DOG_CLASS_ID = 16
dog_boxes = result[DOG_CLASS_ID]
```

### Q: 检测速度太慢怎么办？

A: 尝试以下方法：
1. 确保使用GPU: `--device cuda:0`
2. 降低视频分辨率
3. 使用更小的模型（ResNet-50版本）

### Q: 误检太多怎么办？

A: 提高置信度阈值：`--score-thr 0.7` 或 `0.8`

### Q: 漏检太多怎么办？

A: 降低置信度阈值：`--score-thr 0.3` 或 `0.4`

### Q: 内存不足怎么办？

A:
```bash
# 使用CPU（虽然慢但省显存）
python detect_person_in_video.py input.mp4 --device cpu
```

### Q: 如何获取检测框坐标？

A:
```python
result = inference_detector(model, frame)
person_boxes = result[0]  # shape: [N, 5]

for box in person_boxes:
    x1, y1, x2, y2, score = box
    print(f"人物位置: ({x1:.0f}, {y1:.0f}) - ({x2:.0f}, {y2:.0f}), 置信度: {score:.2f}")
```

---

## 📈 性能参考

| 设备 | 分辨率 | 速度 | 说明 |
|------|--------|------|------|
| RTX 3090 | 1920x1080 | ~15 FPS | ViT-Large模型 |
| RTX 3080 | 1920x1080 | ~12 FPS | ViT-Large模型 |
| RTX 2080Ti | 1280x720 | ~20 FPS | ViT-Large模型 |
| CPU (i9-12900K) | 1280x720 | ~0.5 FPS | 不推荐 |

---

## 📚 进阶学习

想了解更多功能？查看完整教程：
- 📖 [完整教程](PERSON_DETECTION_TUTORIAL.md)
- 🔧 [进阶用法](PERSON_DETECTION_TUTORIAL.md#进阶使用)
- 💻 [API文档](PERSON_DETECTION_TUTORIAL.md#python-api调用)

---

## 🎉 快速测试

使用项目自带的demo视频快速测试：

```bash
# 方法1: 使用命令行工具
python detect_person_in_video.py demo/demo.mp4 \
    --config projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco.py \
    --checkpoint path/to/checkpoint.pth \
    --output demo_result.mp4 \
    --show

# 方法2: 如果已下载模型到checkpoints目录
python detect_person_in_video.py demo/demo.mp4 --show
```

---

## 📞 获取帮助

- 查看帮助信息: `python detect_person_in_video.py --help`
- GitHub Issues: https://github.com/Sense-X/Co-DETR/issues
- 论文: https://arxiv.org/pdf/2211.12860.pdf

---

**祝您使用愉快！🎈**
