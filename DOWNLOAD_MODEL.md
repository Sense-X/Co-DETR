# 模型下载指南

## Hugging Face模型仓库

**仓库地址**: https://huggingface.co/zongzhuofan/co-detr-vit-large-coco

## 文件说明

### 模型文件结构

Hugging Face仓库包含以下文件：
- `pytorch_model.pth` - 模型权重文件 (~2.2GB)
- 其他配置和元数据文件

### 本地配置文件

配置文件在项目中已包含：
- `projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco.py`

## 下载方法

### 方法1: 使用huggingface-cli（推荐）

```bash
# 1. 安装Hugging Face CLI
pip install huggingface-hub

# 2. 下载模型权重
huggingface-cli download zongzhuofan/co-detr-vit-large-coco \
    pytorch_model.pth \
    --local-dir ./checkpoints
```

### 方法2: 使用git clone

```bash
# 1. 安装git-lfs
git lfs install

# 2. 克隆整个仓库
git clone https://huggingface.co/zongzhuofan/co-detr-vit-large-coco ./checkpoints
```

### 方法3: 手动下载

访问 https://huggingface.co/zongzhuofan/co-detr-vit-large-coco/tree/main
下载 `pytorch_model.pth` 到 `./checkpoints/` 目录

## 下载后的目录结构

```
Co-DETR/
├── checkpoints/
│   └── pytorch_model.pth          # 从Hugging Face下载的权重
├── projects/
│   └── configs/
│       └── co_dino_vit/
│           └── co_dino_5scale_vit_large_coco.py  # 本地配置文件
└── detect_person_in_video.py      # 检测脚本
```

## 使用示例

```python
from mmdet.apis import init_detector

# 正确的文件路径
config_file = 'projects/configs/co_dino_vit/co_dino_5scale_vit_large_coco.py'
checkpoint_file = 'checkpoints/pytorch_model.pth'

model = init_detector(config_file, checkpoint_file, device='cuda:0')
```

## 自动下载

我们的脚本支持自动下载：

```bash
# 脚本会自动下载 pytorch_model.pth 到 checkpoints/ 目录
python detect_person_in_video.py your_video.mp4 --download
```

## 验证下载

```bash
# 检查文件是否存在
ls -lh checkpoints/pytorch_model.pth

# 应该显示约2.2GB的文件
```

## 常见问题

### Q: 为什么不下载配置文件？
A: 配置文件已经在项目的 `projects/configs/co_dino_vit/` 目录中，不需要额外下载。

### Q: 可以重命名模型文件吗？
A: 可以，但需要在使用时指定正确的路径。我们建议保持原始文件名 `pytorch_model.pth`。

### Q: 下载失败怎么办？
A:
1. 检查网络连接
2. 尝试使用VPN
3. 使用方法3手动下载
4. 从镜像站点下载（如果有）

## 模型信息

- **模型**: Co-DETR with ViT-Large backbone
- **数据集**: COCO (80 classes)
- **性能**: 66.0 AP on COCO test-dev
- **参数量**: 304M
- **输入尺寸**: 可变（通常1333x800）
