# Panoptic 3D Scene Reconstruction from a Single RGB Image


### [Project Page](https://manuel-dahnert.com/research/panoptic-reconstruction) | [Paper](https://manuel-dahnert.com/static/e9f76636d34de100048d63007b0992b8/dahnert2021panoptic-reconstruction.pdf) | [Arxiv](https://arxiv.org/abs/2111.02444) | [Video](https://www.youtube.com/watch?v=YVxRNHmd5SA)

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

<p align="center">
    <img width="100%" src="teaser-loop.gif"/>
</p>

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
Follow the official instructions to install the [maskrcnn-benchmark repo](https://github.com/facebookresearch/maskrcnn-benchmark).

### Minkowski Engine (fork, custom)
Follow the instructions to compile [our forked Minkowski Engine version](https://github.com/xheon/MinkowskiEngine) from source.

### Compute library
Finally, compile this library. 

```
# Install library
cd lib/csrc/
python setup.py install
```

### 3D-Front pretrained model
Download the pretrained model checkpoint for 3D-Front from https://hidrive.ionos.com/lnk/SeINI9FC and put it into the ``resource`` folder.

## Inference
To run the method on a 3D-Front sample run `python tools/test_nest_single_image.py` with the pre-trained checkpoint.

```
python tools/test_nest_single_image.py -i <path_to_input_image> -o <output_path>
```

## Datasets

### 3D-FRONT [1]

The 3D-FRONT indoor datasets consists of 6,813 furnished apartments.  
We use Blender-Proc [2] to render photo-realistic images from individual rooms.
We use version from 2020-06-14 of the data.

### Download:
We provide the preprocessed 3D-Front data here: https://hidrive.ionos.com/share/nfgac71bo2  .  
Extract the downloaded data into ``data/front3d/`` or adjust the root data path ``lib/config/paths_catalog.py``.  
By downloading our derived work from the original 3D-Front you accept their original Terms of Use (https://gw.alicdn.com/bao/uploaded/TB1ZJUfK.z1gK0jSZLeXXb9kVXa.pdf).


#### Modifications:
- We replace all walls and ceilings and "re-draw" them in order to close holes in the walls, e.g. empty door frames or windows.  
  For the ceiling we use the same geometry as the floor plane to have a closed room envelope.
- We remove following mesh categories: "WallOuter", "WallBottom", "WallTop", "Pocket", "SlabSide", "SlabBottom", "SlabTop",
                                    "Front", "Back", "Baseboard", "Door", "Window", "BayWindow", "Hole", "WallInner", "Beam"
- During rendering, we only render geometry which is assigned to the current room
- We sample each individual (non-empty) room
   - num max tries: 50,000
   - num max samples per room: 30 
- Camera:
   - we fix the camera height at 0.75m and choose a forward-looking camera angle (similar to the original frames in 3D-Front)

#### Structure
- dataset_root/<scene_id>:
  - rgb_<frame_id>.png (320x240x3, color image)
  - depth_<frame_id>.exr (320x240x1, depth image)
  - segmap_<frame_id>.mapped.npz (320x240x2, 2d segmentation with 0: pre-mapped semantics, 1: instances)
  - geometry_<frame_id>.npz (256x256x256x1 distance field at 3cm voxel resolution, pre-truncated at 12 voxels, will be truncated again during data loading)
  - segmentation_<frame_id>.mapped.npz (256x256x256x2 semantic & instance information)
  - weighting_<frame_id>.mapped.npz (256x256x256x1 pre-computed per-voxel weights)
  
In total, we generate 197,352 frames. We filter out frames, which have an inconsistent number of 2D and 3D instances. 
  
For the 3D generation we use a custom C++ pipeline which loads the sampled camera poses, room layout mesh, and the scene objects.
The geometry is cropped to the camera frustum, such that only geometry within the frustum contributes to the DF calculation.
It generates 3D unsigned distance fields at 3cm resolution together with 3D semantic and instance segmentation.
We store the 3D data as sparse grids.


# References

1. Fu et al. - 3d-Front: 3d Furnished Rooms with Layouts and Semantics
1. Denninger et al. - BlenderProc


