# Panoptic 3D Scene Reconstruction from a Single RGB Image


### [Project Page](https://manuel-dahnert.com/research/panoptic-reconstruction) | [Paper](https://manuel-dahnert.com/static/dahnert2021panoptic-reconstruction.pdf) | [Poster](https://manuel-dahnert.com/static/dahnert2021panoptic-reconstruction-poster.pdf) | [Arxiv](https://arxiv.org/abs/2111.02444) | [Video](https://www.youtube.com/watch?v=YVxRNHmd5SA)

> Panoptic 3D Scene Reconstruction from a Single RGB Image <br />
> [Manuel Dahnert](https://manuel-dahnert.com), [Ji Hou](https://sekunde.github.io), [Matthias Nießner](https://niessnerlab.org/members/matthias_niessner/profile.html), [Angela Dai](https://www.3dunderstanding.org/team.html) <br />
> Neural Information Processing Systems (NeurIPS) - 2021


If you find this work useful for your research, please consider citing

```    
@inproceedings{dahnert2021panoptic,
  title={Panoptic 3D Scene Reconstruction From a Single RGB Image},
  author={Dahnert, Manuel and Hou, Ji and Nie{\ss}ner, Matthias and Dai, Angela},
  booktitle={Thirty-Fifth Conference on Neural Information Processing Systems},
  year={2021}
}
```

## Abstract
Understanding 3D scenes from a single image is fundamental to a wide variety of tasks, such as for robotics, 
motion planning, or augmented reality. Existing works in 3D perception from a single RGB image tend to focus on geometric reconstruction only, or geometric reconstruction with semantic segmentation or instance segmentation. Inspired by 2D panoptic segmentation, we propose to unify the tasks of geometric reconstruction, 3D semantic segmentation, and 3D instance segmentation into the task of panoptic 3D scene reconstruction - from a single RGB image, predicting the complete geometric reconstruction of the scene in the camera frustum of the image, along with semantic and instance segmentations. We thus propose a new approach for holistic 3D scene understanding from a single RGB image which learns to lift and propagate 2D features from an input image to a 3D volumetric scene representation. We demonstrate that this holistic view of joint scene reconstruction, semantic, and instance segmentation is beneficial over treating the tasks independently, thus outperforming alternative approaches.

## Environment
The code was tested with the following configuration:
- Ubuntu 20.04
- Python 3.8
- Pytorch 1.7.1
- CUDA 10.2
- Minkowski Engine 0.5.1, fork
- Mask RCNN Benchmark
- Nvidia 2080 Ti

## Installation
```
# Basic conda enviromnent: Creates new conda environment `panoptic`
conda env create --file environment.yaml
conda activate panoptic
```

### MaskRCNN Benchmark
Follow the official instructions to install the [https://github.com/facebookresearch/maskrcnn-benchmark](maskrcnn-benchmark repo).

### Minkowski Engine (fork, custom)
Follow the instructions to compile [https://github.com/xheon/MinkowskiEngine](our forked Minkowski Engine version).


```
# Install library
cd lib/csrc/
python setup.py install
```

