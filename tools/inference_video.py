import os
import sys
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


def main():
    parser = argparse.ArgumentParser(description="DeTRC video inference script")
    parser.add_argument("config", help="path to model config")
    parser.add_argument("checkpoint", help="path to model checkpoint")
    parser.add_argument("video", help="input video file path")
    parser.add_argument("--device", default="cuda:0", help="device for inference")
    args = parser.parse_args()

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

    clip_feats, snippet_nums, total_snippets = extract_features(
        args.video, clip_len, stride_rate, device
    )

    # Prepare inputs for model - DeTRC expects specific format
    raw_feature = clip_feats  # List of tensors, not wrapped in another list
    snippet_num = snippet_nums  # List of integers, not wrapped in another list
    gt_bbox = [np.zeros((0, 3), dtype=np.float32) for _ in clip_feats]  # List of empty arrays
    video_gt_box = [np.zeros((0, 3), dtype=np.float32)]  # Single empty array
    video_meta = [
        {"video_name": os.path.basename(args.video), "origin_snippet_num": total_snippets}
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


if __name__ == "__main__":
    main()
