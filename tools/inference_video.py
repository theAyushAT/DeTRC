import os
import sys
import glob
from shutil import which

# Add current directory to Python path for DeTRC imports
sys.path.append('.')

# Compatibility shim: MMAction2 0.5.0 expects mmcv.utils.CUDA_HOME
# Must be done BEFORE importing mmcv or mmaction
import mmcv
if not hasattr(mmcv.utils, 'CUDA_HOME'):
    cuda_home = os.environ.get('CUDA_HOME')
    if not cuda_home:
        nvcc = which('nvcc')
        if nvcc:
            cuda_home = os.path.dirname(os.path.dirname(nvcc))
    setattr(mmcv.utils, 'CUDA_HOME', cuda_home)

import argparse
import math
from typing import List

import numpy as np
import torch
from mmcv import Config, VideoReader
from mmcv.runner import load_checkpoint
from mmaction.models import build_model
from torchvision import transforms
from torchvision.models import resnet50
from PIL import Image

# Ensure DeTRC model registers with MMAction2 registries before build_model
import DeTRC.model.DetTRC  # noqa: F401


def extract_features(video_path: str, clip_len: int, stride_rate: float, device: str):
    """Extract frame-level features using a ResNet-50 backbone.

    The extracted features are split into clips following the same logic as in
    the dataset preparation. Each feature vector corresponds to one video frame.
    """
    vr = VideoReader(video_path)
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    backbone = resnet50(pretrained=True)
    # remove the final classification layer
    backbone = torch.nn.Sequential(*list(backbone.children())[:-1]).to(device)
    backbone.eval()

    feats: List[torch.Tensor] = []
    with torch.no_grad():
        for frame in vr:
            img = Image.fromarray(frame[:, :, ::-1])  # BGR to RGB
            tensor = transform(img).unsqueeze(0).to(device)
            feat = backbone(tensor).view(-1)
            feats.append(feat.cpu())

    feats = torch.stack(feats) if feats else torch.zeros((0, 2048))
    total_snippets = feats.shape[0]

    # split into clips
    clips: List[torch.Tensor] = []
    snip_nums: List[int] = []
    if total_snippets <= clip_len:
        clips.append(feats)
        snip_nums.append(total_snippets)
    else:
        step = int(clip_len * stride_rate)
        step = max(step, 1)
        clips_num = math.ceil((total_snippets - clip_len) / step) + 1
        for window in range(clips_num):
            start = step * window
            clip = feats[start : start + clip_len]
            clips.append(clip)
            snip_nums.append(clip.shape[0])
    return clips, torch.tensor(snip_nums), total_snippets


def create_visualization_video(video_path, predictions, total_snippets, clip_len, stride_rate, output_video_dir):
    """
    Create a visualization video showing the detected segments overlaid on the original video.
    
    Args:
        video_path: Path to the input video
        predictions: Model predictions for each class
        total_snippets: Total number of frames in the video
        clip_len: Length of each clip
        stride_rate: Stride rate for clip generation
    """
    import cv2
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video writer
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_video_dir, f"{video_name}_detrc_results.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Calculate statistics for each class
    class_stats = []
    for cls_idx, segments in enumerate(predictions):
        if len(segments) > 0:
            total_duration = sum([seg[1] - seg[0] for seg in segments])
            avg_score = sum([seg[2] for seg in segments]) / len(segments)
            class_stats.append({
                'class': cls_idx,
                'count': len(segments),
                'total_duration': total_duration,
                'avg_score': avg_score
            })
    
    # Sort by count (most frequent first)
    class_stats.sort(key=lambda x: x['count'], reverse=True)
    
    # Function to get live count up to current frame
    def get_live_counts(current_frame_idx):
        live_counts = {}
        for cls_idx, segments in enumerate(predictions):
            count = 0
            for seg in segments:
                start_snippet = seg[0] * total_snippets
                end_snippet = seg[1] * total_snippets
                # Count segments that have started by current frame
                if start_snippet <= current_frame_idx:
                    count += 1
            live_counts[cls_idx] = count
        return live_counts
    
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert frame index to snippet index (assuming 1 snippet = 1 frame)
        snippet_idx = frame_idx
        
        # Draw detected segments for this frame
        current_segment_info = None
        for cls_idx, segments in enumerate(predictions):
            for seg in segments:
                start_snippet = seg[0] * total_snippets  # Convert from normalized to snippet index
                end_snippet = seg[1] * total_snippets
                score = seg[2]
                
                # Check if current frame is within this segment
                if start_snippet <= snippet_idx <= end_snippet:
                    current_segment_info = {
                        'class': cls_idx,
                        'start': seg[0],
                        'end': seg[1],
                        'score': score
                    }
                    break
            if current_segment_info:
                break
        
        # Draw current segment info if any
        if current_segment_info:
            # Draw bounding box around the frame (white)
            cv2.rectangle(frame, (10, 10), (width-10, height-10), (255, 255, 255), 3)
            
            # Draw class label and score
            label = f"Class {current_segment_info['class']}: {current_segment_info['score']:.3f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            cv2.rectangle(frame, (10, 10), (10 + label_size[0] + 10, 10 + label_size[1] + 10), (0, 0, 0), -1)
            cv2.putText(frame, label, (15, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Draw segment info
            segment_info = f"Segment: {current_segment_info['start']:.2f}s - {current_segment_info['end']:.2f}s"
            info_size = cv2.getTextSize(segment_info, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(frame, (10, 40), (10 + info_size[0] + 10, 40 + info_size[1] + 10), (0, 0, 0), -1)
            cv2.putText(frame, segment_info, (15, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Get live counts for current frame
        live_counts = get_live_counts(frame_idx)
        
        # Draw class statistics on the right side with live counts
        y_offset = 30
        cv2.putText(frame, "LIVE COUNTS:", (width - 400, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        y_offset += 25
        
        for i, stat in enumerate(class_stats[:5]):  # Show top 5 classes
            cls_idx = stat['class']
            live_count = live_counts.get(cls_idx, 0)
            total_count = stat['count']
            total_duration = stat['total_duration']
            
            # Show live count vs total count
            stat_text = f"Class {cls_idx}: {live_count}/{total_count} segments, {total_duration:.2f}s"
            cv2.putText(frame, stat_text, (width - 400, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            y_offset += 20
        
        # Add frame counter, time, and progress
        progress_percent = (frame_idx / total_snippets) * 100 if total_snippets > 0 else 0
        
        cv2.putText(frame, f"Frame: {frame_idx}/{total_snippets}", (width - 150, y_offset + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Time: {frame_idx/fps:.2f}s", (width - 150, y_offset + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Progress: {progress_percent:.1f}%", (width - 150, y_offset + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Write frame to output video
        out.write(frame)
        frame_idx += 1
    
    # Clean up
    cap.release()
    out.release()
    
    print(f"Visualization video saved to: {output_path}")
    print(f"Total frames processed: {frame_idx}")
    
    # Print summary statistics
    print("\n=== CLASS STATISTICS ===")
    for stat in class_stats:
        print(f"Class {stat['class']}: {stat['count']} segments, Total duration: {stat['total_duration']:.2f}s, Avg score: {stat['avg_score']:.3f}")





def save_results_to_txt(video_path, predictions, output_txt_dir):
    """
    Save detection results to a text file.
    
    Args:
        video_path: Path to the input video
        predictions: Model predictions for each class
        output_txt_dir: Directory to save the text file
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    txt_path = os.path.join(output_txt_dir, f"{video_name}_detrc_results.txt")
    
    with open(txt_path, 'w') as f:
        f.write(f"DeTRC Results for: {os.path.basename(video_path)}\n")
        f.write("=" * 50 + "\n\n")
        
        for cls_idx, segments in enumerate(predictions):
            if len(segments) > 0:
                f.write(f"Class {cls_idx}: {len(segments)} segments\n")
                for seg in segments:
                    f.write(f"  start: {seg[0]:.2f}, end: {seg[1]:.2f}, score: {seg[2]:.4f}\n")
                f.write("\n")
    
    print(f"Results saved to: {txt_path}")


def process_single_video(video_path, model, clip_len, stride_rate, device, output_video_dir, output_txt_dir):
    """
    Process a single video file with DeTRC model.
    
    Args:
        video_path: Path to the input video
        model: Loaded DeTRC model
        clip_len: Length of each clip
        stride_rate: Stride rate for clip generation
        device: Device for inference
        output_video_dir: Directory to save visualization videos
        output_txt_dir: Directory to save text results
    """
    # Extract features
    clip_feats, snippet_nums, total_snippets = extract_features(
        video_path, clip_len, stride_rate, device
    )

    # Prepare inputs for model - DeTRC expects specific format
    raw_feature = clip_feats  # List of tensors, not wrapped in another list
    snippet_num = snippet_nums  # List of integers, not wrapped in another list
    gt_bbox = [np.zeros((0, 3), dtype=np.float32) for _ in clip_feats]  # List of empty arrays
    video_gt_box = [np.zeros((0, 3), dtype=np.float32)]  # Single empty array
    video_meta = [
        {"video_name": os.path.basename(video_path), "origin_snippet_num": total_snippets}
    ]

    with torch.no_grad():
        print(f"Debug - raw_feature structure: {type(raw_feature)}")
        print(f"Debug - raw_feature[0] type: {type(raw_feature[0])}")
        if hasattr(raw_feature[0], 'shape'):
            print(f"Debug - raw_feature[0] shape: {raw_feature[0].shape}")
        print(f"Debug - snippet_num: {snippet_num}")
        print(f"Debug - video_meta: {video_meta}")
        
        outputs = model.forward(
            raw_feature=raw_feature,
            gt_bbox=gt_bbox,
            snippet_num=snippet_num,
            video_gt_box=video_gt_box,
            video_meta=video_meta,
            return_loss=False,
        )

    preds = outputs[0][0]
    for cls_idx, segments in enumerate(preds):
        print(f"Class {cls_idx}: {len(segments)} segments")
        for seg in segments:
            print(f"  start: {seg[0]:.2f}, end: {seg[1]:.2f}, score: {seg[2]:.4f}")
    
    # Save results to text file
    print("\nSaving results to text file...")
    save_results_to_txt(video_path, preds, output_txt_dir)
    
    # Create visualization video
    print("\nCreating visualization video...")
    create_visualization_video(video_path, preds, total_snippets, clip_len, stride_rate, output_video_dir)


def main():
    parser = argparse.ArgumentParser(description="DeTRC video inference script")
    parser.add_argument("config", help="path to model config")
    parser.add_argument("checkpoint", help="path to model checkpoint")
    parser.add_argument("input_path", help="input video file path or directory containing videos")
    parser.add_argument("--device", default="cuda:0", help="device for inference")
    parser.add_argument("--output-video-dir", default="/home/arso/Desktop/exercise_project_ayush/Detr_working/video/output_video", 
                       help="output directory for visualization videos")
    parser.add_argument("--output-txt-dir", default="/home/arso/Desktop/exercise_project_ayush/Detr_working/video/output_txt", 
                       help="output directory for text results")
    args = parser.parse_args()

    # Create output directories
    os.makedirs(args.output_video_dir, exist_ok=True)
    os.makedirs(args.output_txt_dir, exist_ok=True)
    
    device = args.device
    cfg = Config.fromfile(args.config)
    # Remove pretrained parameter if it exists, as DeTRC doesn't support it
    if hasattr(cfg.model, 'pretrained'):
        delattr(cfg.model, 'pretrained')
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.to(device)
    model.eval()

    clip_len = cfg.model.clip_len
    stride_rate = cfg.model.stride_rate

    # Determine if input is a file or directory
    if os.path.isfile(args.input_path):
        # Single video file
        video_files = [args.input_path]
        print(f"Processing single video: {args.input_path}")
    elif os.path.isdir(args.input_path):
        # Directory of videos
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv']
        video_files = []
        for ext in video_extensions:
            video_files.extend(glob.glob(os.path.join(args.input_path, f"*{ext}")))
            video_files.extend(glob.glob(os.path.join(args.input_path, f"*{ext.upper()}")))
        
        if not video_files:
            print(f"No video files found in directory: {args.input_path}")
            return
        print(f"Found {len(video_files)} video files to process")
    else:
        print(f"Error: {args.input_path} is neither a file nor a directory")
        return

    # Process each video
    for video_path in video_files:
        print(f"\n{'='*60}")
        print(f"Processing: {os.path.basename(video_path)}")
        print(f"{'='*60}")
        
        try:
            process_single_video(video_path, model, clip_len, stride_rate, device, args.output_video_dir, args.output_txt_dir)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
            continue
    
    print(f"\n{'='*60}")
    print("All videos processed!")
    print(f"Output videos: {args.output_video_dir}")
    print(f"Output text files: {args.output_txt_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
