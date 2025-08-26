# DeTRC Environment Setup Guide

## System Requirements
- **OS**: Linux (Ubuntu 20.04+ recommended)
- **GPU**: NVIDIA GPU with CUDA support (tested with RTX 3090)
- **CUDA Driver**: 535.230.02+
- **CUDA Runtime**: 12.1+
- **Memory**: 16GB+ RAM recommended

## Environment Information
- **Python Version**: 3.9
- **PyTorch Version**: 2.5.1
- **CUDA Version**: 12.1
- **cuDNN Version**: 8.9.2.26

## Quick Setup

### 1. Create Conda Environment
```bash
conda env create -f detrc_environment.yml
```

### 2. Activate Environment
```bash
conda activate detrc
```

### 3. Install DeTRC
```bash
cd /path/to/DeTRC
pip install -e .
```

## Manual Setup (if conda env creation fails)

### 1. Create Base Environment
```bash
conda create -n detrc python=3.9
conda activate detrc
```

### 2. Install PyTorch with CUDA
```bash
conda install pytorch=2.5.1 torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

### 3. Install CUDA Toolkit
```bash
conda install cudatoolkit=12.1 cudnn=8.9.2.26 -c conda-forge
```

### 4. Install MMCV and MMAction2
```bash
pip install mmcv-full==1.7.2
pip install mmaction2==0.24.1
```

### 5. Install Other Dependencies
```bash
conda install numpy opencv pillow matplotlib scipy scikit-learn pandas h5py -c conda-forge
pip install terminaltables opencv-python opencv-contrib-python
```

## Verification

### Check CUDA
```bash
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda}')"
```

### Check MMAction2
```bash
python -c "import mmaction; print(f'MMAction2 version: {mmaction.__version__}')"
```

## Troubleshooting

### Common Issues:
1. **CUDA version mismatch**: Ensure PyTorch CUDA version matches system CUDA
2. **MMCV/MMAction2 conflicts**: Use exact versions specified
3. **Permission errors**: Use `sudo` for system-wide installations

### Recreate Environment:
```bash
conda deactivate
conda env remove -n detrc
conda env create -f detrc_environment.yml
```

## Usage
```bash
conda activate detrc
python tools/inference_video.py config_file checkpoint video_path
```
