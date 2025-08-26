# Efficient Action Counting with Dynamic Queries

<p align="center">
<a href="https://arxiv.org/abs/2403.01543v3", target="_blank">
<img src="https://img.shields.io/static/v1?style=for-the-badge&message=arXiv&color=B31B1B&logo=arXiv&logoColor=FFFFFF&label="></a>
<a href="https://shirleymaxx.github.io/DeTRC/", target="_blank">
<img src="https://img.shields.io/badge/_-Project-18405a?style=for-the-badge&logo=Google%20Chrome&logoColor=white" alt="Project Page"></a>
<a href="https://www.youtube.com/watch?v=j98Zriek2xU&ab_channel=XiaoxuanMa", target="_blank">
<img src="https://img.shields.io/badge/_-Video-ea3323?style=for-the-badge&logo=Youtube&logoColor=white" alt="YouTube"></a>
</p>


This is the official PyTorch implementation of the paper "Efficient Action Counting with Dynamic Queries". It provides a novel perspective to tackle the *Temporal Repetition Counting* problem using a simple yet effective representation for action cycles, reducing the computational complexity from **quadratic** to **linear** with SOTA performance.

<p align="center">
<img src="https://shirleymaxx.github.io/DeTRC/images/structure_diff.jpg" style="width: 100%;">
</p>


## Installation
We build our code based on the MMaction2 project (1.3.10 version). See [here](https://github.com/open-mmlab/mmaction2) for more details if you are interested. Install the runtime dependencies with:
```shell
pip install -r requirements.txt
```
If you need a specific CUDA or PyTorch build for `mmcv-full`, please refer to the [mmcv](https://github.com/open-mmlab/mmcv) documentation.

Then, our code can be built by
```shell
cd DeTRC
pip3 install -e .
```

Then, Install the 1D Grid Sampling and RoI Align operators.
```shell
cd DeTRC/model
python setup.py build_ext --inplace
```

## Data
We use the TSN feature of RepCountA and UCFRep datasets. Please refer to the guidance [here](./docs/prepare_dataset_DeTRC.md).

## Train

Our model can be trained with

```python
python tools/train.py DeTRC/configs/repcount_tsn_feature_enc_contrastive.py --validate
```

We recommend to set the `--validate` flag to monitor the training process.

## Test
If you want to test the pretrained model, please use the following code. We provide the pretrained model [here](https://pan.baidu.com/s/1M1qOgytY87KPFOthKpIUDw?pwd=awxe).
```shell
python tools/test.py DeTRC/configs/repcount_tsn_feature_enc_contrastive.py PATH_TO_CHECKPOINT
```

## Results

| ![pose_1](https://shirleymaxx.github.io/DeTRC/images/assets/stu7_8.gif) | ![pose_1](https://shirleymaxx.github.io/DeTRC/images/assets/stu1_27.gif) |
| ------------------------------------------------------------ | ------------------------------------------------------------ |

Comparison with SOTA:

![cmp_1](https://shirleymaxx.github.io/DeTRC/images/assets/sota2.gif)

## Citation

If you find our work useful for your project, please cite the paper as below:

```
@article{li2024efficient,
  title={Efficient Action Counting with Dynamic Queries},
  author={Li, Zishi and Ma, Xiaoxuan and Shang, Qiuyan and Zhu, Wentao and Ci, Hai and Qiao, Yu and Wang, Yizhou},
  journal={arXiv preprint arXiv:2403.01543},
  year={2024}
}
```

