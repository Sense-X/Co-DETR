#!/usr/bin/env python
"""
使用Co-DETR ViT-Large模型进行视频中的人物检测
支持从Hugging Face下载模型或使用本地模型
"""
import argparse
import os
import cv2
import mmcv
import torch
from mmdet.apis import inference_detector, init_detector
from projects import *


def download_model_from_huggingface(save_dir='./checkpoints'):
    """从Hugging Face下载Co-DETR ViT-Large模型"""
    import subprocess

    os.makedirs(save_dir, exist_ok=True)

    model_repo = "zongzhuofan/co-detr-vit-large-coco"

    # 下载模型权重
    checkpoint_path = os.path.join(save_dir, "co_dino_5scale_vit_large_coco.pth")
    config_path = os.path.join(save_dir, "co_dino_5scale_vit_large_coco.py")

    if not os.path.exists(checkpoint_path):
        print(f"正在从Hugging Face下载模型权重...")
        cmd = f"huggingface-cli download {model_repo} co_dino_5scale_vit_large_coco.pth --local-dir {save_dir}"
        subprocess.run(cmd, shell=True, check=True)
        print(f"模型权重已下载到: {checkpoint_path}")

    if not os.path.exists(config_path):
        print(f"正在下载配置文件...")
        cmd = f"huggingface-cli download {model_repo} co_dino_5scale_vit_large_coco.py --local-dir {save_dir}"
        subprocess.run(cmd, shell=True, check=True)
        print(f"配置文件已下载到: {config_path}")

    return config_path, checkpoint_path


def filter_person_results(results, score_threshold=0.3):
    """
    过滤检测结果，只保留人物检测框

    Args:
        results: 模型输出结果
        score_threshold: 置信度阈值

    Returns:
        filtered_results: 只包含人物的检测结果
    """
    # COCO数据集中，person类别的索引是0
    PERSON_CLASS_ID = 0

    # 创建一个新的结果列表
    filtered_results = [[] for _ in range(len(results))]

    # 只保留person类的检测框
    if len(results[PERSON_CLASS_ID]) > 0:
        person_boxes = results[PERSON_CLASS_ID]
        # 过滤低置信度的框
        person_boxes = person_boxes[person_boxes[:, -1] > score_threshold]
        filtered_results[PERSON_CLASS_ID] = person_boxes

    return filtered_results


def detect_person_in_video(video_path, config_path, checkpoint_path,
                          output_path=None, score_threshold=0.5,
                          device='cuda:0', show_video=False):
    """
    在视频中检测人物

    Args:
        video_path: 输入视频路径
        config_path: 模型配置文件路径
        checkpoint_path: 模型权重文件路径
        output_path: 输出视频路径
        score_threshold: 检测置信度阈值
        device: 运行设备 ('cuda:0' 或 'cpu')
        show_video: 是否实时显示检测结果
    """
    # 初始化模型
    print(f"正在加载模型...")
    print(f"配置文件: {config_path}")
    print(f"权重文件: {checkpoint_path}")
    print(f"设备: {device}")

    model = init_detector(config_path, checkpoint_path, device=device)
    print("模型加载完成!")

    # 打开视频
    video_reader = mmcv.VideoReader(video_path)
    print(f"\n视频信息:")
    print(f"  分辨率: {video_reader.width}x{video_reader.height}")
    print(f"  帧率: {video_reader.fps}")
    print(f"  总帧数: {len(video_reader)}")

    # 准备视频写入器
    video_writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            output_path, fourcc, video_reader.fps,
            (video_reader.width, video_reader.height))
        print(f"\n输出视频: {output_path}")

    print(f"\n开始检测人物 (置信度阈值: {score_threshold})...\n")

    # 统计信息
    total_detections = 0
    frame_count = 0

    # 逐帧处理
    for frame in mmcv.track_iter_progress(video_reader):
        frame_count += 1

        # 推理
        result = inference_detector(model, frame)

        # 只保留人物检测结果
        person_result = filter_person_results(result, score_threshold)

        # 统计当前帧的人数
        num_persons = len(person_result[0])
        total_detections += num_persons

        # 可视化结果
        # 修改类别名称只显示"person"
        model.CLASSES = ('person',) + model.CLASSES[1:]
        frame_with_boxes = model.show_result(
            frame, person_result,
            score_thr=score_threshold,
            bbox_color=(0, 255, 0),  # 绿色边界框
            text_color=(0, 255, 0)   # 绿色文字
        )

        # 在画面上添加检测统计信息
        info_text = f"Frame: {frame_count} | Persons: {num_persons} | Threshold: {score_threshold}"
        cv2.putText(frame_with_boxes, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # 显示视频
        if show_video:
            cv2.namedWindow('Person Detection', cv2.WINDOW_NORMAL)
            cv2.imshow('Person Detection', frame_with_boxes)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n用户中断检测")
                break

        # 写入输出视频
        if video_writer:
            video_writer.write(frame_with_boxes)

    # 清理资源
    if video_writer:
        video_writer.release()

    if show_video:
        cv2.destroyAllWindows()

    # 输出统计信息
    print(f"\n检测完成!")
    print(f"总帧数: {frame_count}")
    print(f"平均每帧检测到的人数: {total_detections/frame_count:.2f}")

    if output_path:
        print(f"结果已保存到: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Co-DETR视频人物检测工具')
    parser.add_argument('video', help='输入视频文件路径')
    parser.add_argument('--config', help='模型配置文件路径 (可选，会自动从HF下载)')
    parser.add_argument('--checkpoint', help='模型权重文件路径 (可选，会自动从HF下载)')
    parser.add_argument('--output', '-o', help='输出视频文件路径')
    parser.add_argument('--score-thr', type=float, default=0.5,
                       help='检测置信度阈值 (默认: 0.5)')
    parser.add_argument('--device', default='cuda:0',
                       help='运行设备 (默认: cuda:0)')
    parser.add_argument('--show', action='store_true',
                       help='实时显示检测结果')
    parser.add_argument('--download', action='store_true',
                       help='从Hugging Face下载模型')

    args = parser.parse_args()

    # 检查是否需要下载模型
    if args.download or (not args.config or not args.checkpoint):
        print("正在从Hugging Face下载模型...")
        config_path, checkpoint_path = download_model_from_huggingface()
    else:
        config_path = args.config
        checkpoint_path = args.checkpoint

    # 检查视频文件是否存在
    if not os.path.exists(args.video):
        raise FileNotFoundError(f"视频文件不存在: {args.video}")

    # 设置默认输出路径
    if not args.output:
        video_name = os.path.splitext(os.path.basename(args.video))[0]
        args.output = f"{video_name}_person_detection.mp4"

    # 执行检测
    detect_person_in_video(
        video_path=args.video,
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        output_path=args.output,
        score_threshold=args.score_thr,
        device=args.device,
        show_video=args.show
    )


if __name__ == '__main__':
    main()
