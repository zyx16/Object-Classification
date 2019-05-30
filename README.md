# Media and Cognition Course Porject

Pascal VOC 2012 Object Classification and Weakly Supervised Object Detection(WSOD)

## Method

### Multi-label classification

[Multi-Label Image Recognition with Graph Convolutional Networks](https://arxiv.org/abs/1904.03582)

### WSOD

[Class activation map](https://arxiv.org/pdf/1512.04150.pdf) and [Hide and Seek](https://arxiv.org/pdf/1704.04232)

## Dependencies

- Python 3.6
- Pytorch 1.0.1
- NVIDIA GPU + CUDA
- Python packages: numpy, opencv-python, tensorboardX(optional)

## Data

### Dataset

Pascal VOC dataset

Load from data/\*\_anno.txt. You can generate your own annotation file according to the given ones.

### Others

voc adjacent matrix and voc world vector generated from data in [ML\_GCN](https://github.com/chenzhaomin123/ML_GCN).

all in data/

## Usage

### Train

#### Multi-label classification

`python train.py -opt option/train_densenet161_gcn.json`

#### WSOD

`python train.py -opt option/train_densenet161_HaS.json`

### Test

#### Multi-label classification

`python test.py -opt option/test_densenet161_gcn.json`

#### WSOD

- firt, generate your own ground truth files and put them into mAP/input/ground-truth according to README in mAP
- `python test_WSOD.py -opt option/test_densenet161_WSOD.json`

## Acknowledgement

- Code architecture is inspired by [BasicSR](https://github.com/xinntao/BasicSR)
- mAP from [Cartucho/mAP](https://github.com/Cartucho/mAP)
