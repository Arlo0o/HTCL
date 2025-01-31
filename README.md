## [ECCV 2024] Hierarchical Temporal Context Learning for Camera-based Semantic Scene Completion



 [![arXiv paper](https://img.shields.io/badge/arXiv%20%2B%20supp-2407.02077-purple)](https://arxiv.org/abs/2407.02077) 
[![Code page](https://img.shields.io/badge/Project%20Page-HTCL-red)](https://github.com/Arlo0o/HTCL/)


### Demo:
<div align=center><img width="640" height="360" src="./assets/teaser.gif"/></div>


### Framework:
<div align=center><img width="640" height="360" src="./assets/overall.png"/></div>


### Abstract:
Camera-based 3D semantic scene completion (SSC) is pivotal for predicting complicated 3D layouts with limited 2D image observations. The existing mainstream solutions generally leverage temporal information by roughly stacking history frames to supplement the current frame, such straightforward temporal modeling inevitably diminishes valid clues and increases learning difficulty. To address this problem, we present HTCL, a novel Hierarchical Temporal Context Learning paradigm for improving camera-based semantic scene completion.
The primary innovation of this work involves decomposing temporal context learning into two hierarchical steps: (a) cross-frame affinity measurement and (b) affinity-based dynamic refinement. Firstly, to separate critical relevant context from redundant information, we introduce the pattern affinity with scale-aware isolation and multiple independent learners for fine-grained contextual correspondence modeling. Subsequently, to dynamically compensate for incomplete observations, we adaptively refine the feature sampling locations based on initially identified locations with high affinity and their neighboring relevant regions. Our method ranks $1^{st}$ on the SemanticKITTI benchmark and even surpasses LiDAR-based methods in terms of mIoU on the OpenOccupancy benchmark.



# Table of Content
- [News](#news)
- [Quick Start](#quick-installation-on-a100)
- [Installation](#step-by-step-installation-instructions)
- [Prepare Data](#prepare-data)
- [Pretrained Model](#pretrained-model)
- [Training & Evaluation](#training--evaluation)
- [Visualization](#visualization)
- [License](#license)
- [Acknowledgements](#acknowledgements)


# News
- [2023/07]: Demo and code released.
- [2023/07]: Paper is on [arxiv](https://arxiv.org/abs/2407.02077).
- [2023/07]: Paper is accepted on ECCV 2024.
- [2024/01]: Update visualization tools.

# Quick Installation on A100

You can use our pre-picked environment on NVIDIA A100 with the following steps if using the same hardware:

**a. Download the pre-picked package:  [occA100](https://drive.google.com/file/d/1JX1TM13yGLjvfz54pTZ4so2nFPNcYa0h/view?usp=sharing).**

**b. Unpack environment into directory occA100**.
```shell
cd /opt/conda/envs/
mkdir -p occA100
tar -xzf occA100.tar.gz -C occA100 
```
**c. Activate the environment. This adds occA100/bin to your path.**
```shell
source occA100/bin/activate
```

You can also use Python executable file without activating or fixing the prefixes. 
```shell
./occA100/bin/python
```


# Step-by-step Installation Instructions

Following https://mmdetection3d.readthedocs.io/en/latest/getting_started.html#installation

**a. Create a conda virtual environment and activate it.**
python > 3.7 may not be supported, because installing open3d-python with py>3.7 causes errors.
```shell
conda create -n occupancy python=3.7 -y
conda activate occupancy
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
conda install pytorch==1.10.1 torchvision==0.11.2 torchaudio==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
```

**c. Install gcc>=5 in conda env (optional).**
I do not use this step.
```shell
conda install -c omgarcia gcc-6 # gcc-6.2
```

**c. Install mmcv-full.**
```shell
pip install mmcv-full==1.4.0
```

**d. Install mmdet and mmseg.**
```shell
pip install mmdet==2.14.0
pip install mmsegmentation==0.14.1
```

**e. Install mmdet3d from source code.**
```shell
cd mmdetection3d
git checkout v0.17.1 # Other versions may not be compatible.
python setup.py install
```
Please check your CUDA version for [mmdet3d](https://github.com/open-mmlab/mmdetection3d/issues/2427) if encountered import problem. 

**f. Install other dependencies.**
```shell
pip install timm
pip install open3d-python
pip install PyMCubes
```


## Known problems

### AttributeError: module 'distutils' has no attribute 'version'
The error appears due to the version of "setuptools", try:
```shell
pip install setuptools==59.5.0
```




# Prepare Data

- **a. You need to download**

     - The **Odometry calibration** (Download odometry data set (calibration files)) and the **RGB images** (Download odometry data set (color)) from [KITTI Odometry website](http://www.cvlibs.net/datasets/kitti/eval_odometry.php), extract them to the folder `data/occupancy/semanticKITTI/RGB/`.
     - The **Velodyne point clouds** (Download [data_odometry_velodyne](http://www.cvlibs.net/download.php?file=data_odometry_velodyne.zip)) and the **SemanticKITTI label data** (Download [data_odometry_labels](http://www.semantic-kitti.org/assets/data_odometry_labels.zip)) for sparse LIDAR supervision in training process, extract them to the folders ``` data/lidar/velodyne/ ``` and ``` data/lidar/lidarseg/ ```, separately. 


- **b. Prepare KITTI voxel label (see sh file for more details)**
```
bash process_kitti.sh
```




# Pretrained Model

Download [Pretrained model](https://drive.google.com/file/d/1e6QWM__dhN5Bvme76MErkwJtD_UHQH06/view?usp=sharing) on SemanticKITTI and [Efficientnet-b7 pretrained model](https://drive.google.com/file/d/14_Qci68SG-g-9BRwgR420cseH6BCeaSp/view?usp=sharing), put them in the folder `./pretrain`.




# Training & Evaluation

## Single GPU
- **Train with single GPU:**
```
export PYTHONPATH="."  
python tools/train.py   \
            projects/configs/occupancy/semantickitti/temporal_baseline.py
```

- **Evaluate with single GPUs:**
```
export PYTHONPATH="."  
bash  run_eval_kitti.sh   \
            projects/configs/occupancy/semantickitti/temporal_baseline.py \
            pretrain/pretrain.pth 
```




## Multiple GPUS
- **Train with n GPUs:**
```
bash run.sh  \
        projects/configs/occupancy/semantickitti/temporal_baseline.py  n
```

- **Evaluate with n GPUs:**
```
 bash tools/dist_test.sh  \
            projects/configs/occupancy/semantickitti/temporal_baseline.py \
            pretrain/pretrain.pth  n
```

## Visualization

We use mayavi to visualize the predictions. Please install [mayavi](https://docs.enthought.com/mayavi/mayavi/installation.html) following the official installation instruction. Then, use the following commands to visualize the outputs.


```
export PYTHONPATH="."  
python tools/save_vis.py projects/configs/occupancy/semantickitti/stereoscene.py \
            pretrain/pretrain_stereoscene.pth  --eval mAP
python tools/visualization.py
```



# License
This repository is released under the Apache 2.0 license as found in the [LICENSE](LICENSE) file.



# Acknowledgements
Many thanks to these excellent open source projects: 
- [MonoScene](https://github.com/astra-vision/MonoScene)
- [StereoScene](https://github.com/Arlo0o/StereoScene)



## Citation
If you find our paper and code useful for your research, please consider citing:

```bibtex
@article{li2024hierarchical,
  title={Hierarchical Temporal Context Learning for Camera-based Semantic Scene Completion},
  author={Li, Bohan and Deng, Jiajun and Zhang, Wenyao and Liang, Zhujin and Du, Dalong and Jin, Xin and Zeng, Wenjun},
  journal={arXiv preprint arXiv:2407.02077},
  year={2024}
}

```

