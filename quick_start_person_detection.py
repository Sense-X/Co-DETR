#!/usr/bin/env python
"""
Co-DETR 视频人物检测 - 快速入门版本
最简单的使用方式
"""
import cv2
import mmcv
from mmdet.apis import inference_detector, init_detector
from projects import *


def main():
    # ============ 配置区域 ============
    # 1. 设置输入视频路径
    VIDEO_PATH = 'demo/demo.mp4'  # 修改为你的视频路径

    # 2. 设置模型路径（从Hugging Face下载后的路径）
    CONFIG_FILE = 'checkpoints/co_dino_5scale_vit_large_coco.py'
    CHECKPOINT_FILE = 'checkpoints/co_dino_5scale_vit_large_coco.pth'

    # 3. 设置输出视频路径
    OUTPUT_VIDEO = 'output_person_detection.mp4'

    # 4. 设置检测参数
    SCORE_THRESHOLD = 0.5  # 置信度阈值 (0-1之间，越高越严格)
    DEVICE = 'cuda:0'      # 'cuda:0' 使用GPU, 'cpu' 使用CPU

    # 5. 是否实时显示检测结果
    SHOW_VIDEO = True      # True显示, False不显示（后台处理）
    # ==================================

    print("=" * 60)
    print("Co-DETR 视频人物检测")
    print("=" * 60)

    # 步骤1: 加载模型
    print("\n[1/4] 正在加载模型...")
    model = init_detector(CONFIG_FILE, CHECKPOINT_FILE, device=DEVICE)
    print("✓ 模型加载完成")

    # 步骤2: 打开视频
    print(f"\n[2/4] 正在读取视频: {VIDEO_PATH}")
    video_reader = mmcv.VideoReader(VIDEO_PATH)
    print(f"✓ 视频信息: {video_reader.width}x{video_reader.height}, "
          f"{video_reader.fps} FPS, 共{len(video_reader)}帧")

    # 步骤3: 准备输出视频
    print(f"\n[3/4] 准备输出视频: {OUTPUT_VIDEO}")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(
        OUTPUT_VIDEO, fourcc, video_reader.fps,
        (video_reader.width, video_reader.height)
    )

    # 步骤4: 逐帧处理
    print(f"\n[4/4] 开始检测人物 (阈值={SCORE_THRESHOLD})...")
    print("-" * 60)

    total_persons = 0
    frame_count = 0

    for frame in mmcv.track_iter_progress(video_reader):
        frame_count += 1

        # 检测
        result = inference_detector(model, frame)

        # COCO数据集中person类别索引为0
        person_boxes = result[0]  # shape: [N, 5] (x1, y1, x2, y2, score)

        # 过滤低置信度
        person_boxes = person_boxes[person_boxes[:, -1] > SCORE_THRESHOLD]
        num_persons = len(person_boxes)
        total_persons += num_persons

        # 只显示person类的检测结果
        person_only_result = [[] for _ in range(len(result))]
        person_only_result[0] = person_boxes

        # 可视化
        frame_with_boxes = model.show_result(
            frame, person_only_result,
            score_thr=SCORE_THRESHOLD,
            bbox_color=(0, 255, 0),
            text_color=(0, 255, 0)
        )

        # 添加信息文字
        info_text = f"Frame: {frame_count}/{len(video_reader)} | Persons: {num_persons}"
        cv2.putText(frame_with_boxes, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # 显示
        if SHOW_VIDEO:
            cv2.imshow('Person Detection (按q退出)', frame_with_boxes)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n用户中断")
                break

        # 保存
        video_writer.write(frame_with_boxes)

    # 清理
    video_writer.release()
    cv2.destroyAllWindows()

    # 输出统计
    print("\n" + "=" * 60)
    print("检测完成!")
    print(f"总帧数: {frame_count}")
    print(f"平均每帧人数: {total_persons/frame_count:.2f}")
    print(f"输出视频: {OUTPUT_VIDEO}")
    print("=" * 60)


if __name__ == '__main__':
    """
    使用步骤：
    1. 从Hugging Face下载模型：
       huggingface-cli download zongzhuofan/co-detr-vit-large-coco \\
           co_dino_5scale_vit_large_coco.pth \\
           co_dino_5scale_vit_large_coco.py \\
           --local-dir ./checkpoints

    2. 修改上面的配置区域（VIDEO_PATH等）

    3. 运行脚本：
       python quick_start_person_detection.py
    """
    main()
