# SelfDM


## Citation

The code is for **Research Use Only**. If you use this code for your research, please consider citing:

```
@article{liuyaqi2022tip,
  title={Two-Stage Copy-Move Forgery Detection With Self Deep Matching and Proposal SuperGlue},
  author={Liu, Yaqi and Xia, Chao and Zhu, Xiaobin and Xu, Shengwei},
  journal={{IEEE} Transactions on Image Processing},
  volume={31},
  pages={541--555},
  year={2022},
  publisher={IEEE}
}
```

## Acknowledgement

Our project relies on the codes from [DeepMask](https://github.com/foolwood/deepmask-pytorch), [SuperGlue](https://github.com/magicleap/SuperGluePretrainedNetwork), and [ConvCRF](https://github.com/MarvinTeichmann/ConvCRF?tab=readme-ov-file). You can download their pretrained models from their projects and kindly cite their papers.

## Getting Started


The code was written and supported by [Yaqi Liu](https://github.com/yaqiliu-cs).

### Prerequisites

- Linux
- Python 3
- NVIDIA GPU + CUDA

### Installation

```bash
conda create --name selfdm python=3.6 -y 
conda activate selfdm
conda install pytorch=1.7.1 torchvision=0.8.2 torchaudio=0.7.2 cudatoolkit=11.0 -c pytorch
pip install -r requirements.txt 
```
- Clone this repo:
```bash
git clone https://github.com/yaqiliu-cs/SelfDM-TIP
cd SelfDM-TIP
```

### Generated BESTI dataset
- [BESTI](https://drive.google.com/file/d/1lkSk0YXKF5lQ7Byytjs2jEz-w74XNRn0/view?usp=sharing).

### Pre-trained model
Download our pre-trained [selfdm](https://drive.google.com/file/d/16mDQmmNlLRj2TC-td2NiczpFc3WX1GhE/view?usp=sharing)

- Test selfdm model
```bash
python proposals_coco.py
```

## Authorship

**Yaqi Liu**

Beijing Electronic Science and Technology Institute, Beijing 100070, China

E-mail: liuyaqi@besti.edu.cn

**Yifan Zhang**

Beijing Electronic Science and Technology Institute, Beijing 100070, China

E-mail: zhang1fan2000@163.com
