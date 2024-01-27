import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from typing import Dict, Any

from lib import modeling

import lib.data.transforms2d as t2d
from lib.config import config
from lib.utils.intrinsics import adjust_intrinsic
from lib.structures import DepthMap

import lib.visualize as vis
from lib.visualize.utils import get_class_labels
from lib.visualize.image import write_detection_image, write_depth, get_image_for_instance
from lib.structures.frustum import compute_camera2frustum_transform

import trimesh
from skimage import measure 
from pysdf import SDF
import kaolin as kal

def main(opts):
    output_path = Path(opts.output)
    output_path.mkdir(exist_ok=True, parents=True)
    
    instance_ids, instance_sdfs, cropped_instance_images, cropped_instance_masks, instance_texts = run_panoptic(opts, output_path)
    
    run_sdfusion([{'id': id, 'sdf': sdf, 'img': img, 'mask':mask, 'text': text} for id, sdf, img, mask, text in zip(instance_ids, instance_sdfs, cropped_instance_images, cropped_instance_masks, instance_texts)], output_path)

def run_panoptic(opts, output_path):
    configure_inference(opts)

    device = torch.device("cuda:0")

    # Define model and load checkpoint.
    print("Load model...")
    model = modeling.PanopticReconstruction()
    checkpoint = torch.load(opts.model)
    model.load_state_dict(checkpoint["model"])  # load model checkpoint
    model = model.to(device)  # move to gpu
    model.switch_test()
    
    # SDFusion model

    # Define image transformation.
    color_image_size = (320, 240)
    depth_image_size = (160, 120)

    imagenet_stats = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    image_transforms = t2d.Compose([
        t2d.Resize(color_image_size),
        t2d.ToTensor(),
    ])
    
    image_transforms_normalize = t2d.Compose([
        t2d.Normalize(imagenet_stats[0], imagenet_stats[1]),  # use imagenet stats to normalize image
    ])


    # Open and prepare input image.
    print("Load input image...")
    sdfusion_input_image = image_transforms(Image.open(opts.input).convert('RGB'))
    input_image = image_transforms(Image.open(opts.input))
    input_image = image_transforms_normalize(input_image)
    input_image = input_image.unsqueeze(0).to(device)

    # Prepare intrinsic matrix.
    front3d_intrinsic = np.array(config.MODEL.PROJECTION.INTRINSIC)
    front3d_intrinsic = adjust_intrinsic(front3d_intrinsic, color_image_size, depth_image_size)
    front3d_intrinsic = torch.from_numpy(front3d_intrinsic).to(device).float()

    # Prepare frustum mask.
    front3d_frustum_mask = np.load(str("data/front3d/frustum_mask.npz"))["mask"]
    front3d_frustum_mask = torch.from_numpy(front3d_frustum_mask).bool().to(device).unsqueeze(0).unsqueeze(0)

    print("Perform panoptic 3D scene reconstruction...")
    with torch.no_grad():
        results = model.inference(input_image, front3d_intrinsic, front3d_frustum_mask)
        
        occupancy_instances = extract_instance_occupancies(results, output_path)
        instance_sdfs, instance_ids = occupancies_to_sdfs(occupancy_instances, output_path)
        
        cropped_instance_images, cropped_instance_masks = get_image_for_instance(sdfusion_input_image, results["instance"], output_path)
        instance_texts = instance_ids_to_labels(instance_ids, results)
        
    print(f"Visualize results, save them at {output_path}")
    visualize_results(results, output_path)
    
    with open(output_path / "semantic_labels.json", "w") as file:
        json.dump(dict(zip(instance_ids, instance_texts)), file, indent=4)
    
    return instance_ids, instance_sdfs, cropped_instance_images, cropped_instance_masks, instance_texts
    
def configure_inference(opts):
    # load config
    config.OUTPUT_DIR = opts.output
    config.merge_from_file(opts.config_file)
    config.merge_from_list(opts.opts)
    # inference settings
    config.MODEL.FRUSTUM3D.IS_LEVEL_64 = False
    config.MODEL.FRUSTUM3D.IS_LEVEL_128 = False
    config.MODEL.FRUSTUM3D.IS_LEVEL_256 = False
    config.MODEL.FRUSTUM3D.FIX = True


def visualize_results(results: Dict[str, Any], output_path: os.PathLike) -> None:
    device = results["input"].device

    # Visualize depth prediction
    depth_map: DepthMap = results["depth"]
    depth_map.to_pointcloud(output_path / "depth_prediction.ply")
    write_depth(depth_map, output_path / "depth_map.png")

    # Visualize 2D detections
    #write_detection_image(results["input"], results["instance"], output_path / "detection.png")

    # Visualize projection
    vis.write_pointcloud(results["projection"].C[:, 1:], None, output_path / "projection.ply")

    # Visualize 3D outputs
    dense_dimensions = torch.Size([1, 1] + config.MODEL.FRUSTUM3D.GRID_DIMENSIONS)
    min_coordinates = torch.IntTensor([0, 0, 0]).to(device)
    truncation = config.MODEL.FRUSTUM3D.TRUNCATION
    iso_value = config.MODEL.FRUSTUM3D.ISO_VALUE

    geometry = results["frustum"]["geometry"]
    surface, _, _ = geometry.dense(dense_dimensions, min_coordinates, default_value=truncation)
    instances = results["panoptic"]["panoptic_instances"]
    semantics = results["panoptic"]["panoptic_semantics"]

    # Main outputs
    camera2frustum = compute_camera2frustum_transform(depth_map.intrinsic_matrix.cpu(), torch.tensor(results["input"].size()) / 2.0,
                                                      config.MODEL.PROJECTION.DEPTH_MIN,
                                                      config.MODEL.PROJECTION.DEPTH_MAX,
                                                      config.MODEL.PROJECTION.VOXEL_SIZE)


    # remove padding: original grid size: [256, 256, 256] -> [231, 174, 187]
    camera2frustum[:3, 3] += (torch.tensor([256, 256, 256]) - torch.tensor([231, 174, 187])) / 2
    frustum2camera = torch.inverse(camera2frustum)
    print(frustum2camera)
    vis.write_distance_field(surface.squeeze(), None, output_path / "mesh_geometry.ply", transform=frustum2camera)
    vis.write_distance_field(surface.squeeze(), instances.squeeze(), output_path / "mesh_instances.ply", transform=frustum2camera)
    vis.write_distance_field(surface.squeeze(), semantics.squeeze(), output_path / "mesh_semantics.ply", transform=frustum2camera)

    with open(output_path / "semantic_classes.json", "w") as f:
        json.dump(results["panoptic"]["panoptic_semantic_mapping"], f, indent=4)

    # Visualize auxiliary outputs
    vis.write_pointcloud(geometry.C[:, 1:], None, output_path / "sparse_coordinates.ply")

    surface_mask = surface.squeeze() < iso_value
    points = surface_mask.squeeze().nonzero()
    point_semantics = semantics[surface_mask]
    point_instances = instances[surface_mask]

    vis.write_pointcloud(points, None, output_path / "points_geometry.ply")
    vis.write_semantic_pointcloud(points, point_semantics, output_path / "points_surface_semantics.ply")
    vis.write_semantic_pointcloud(points, point_instances, output_path / "points_surface_instances.ply")

def extract_instance_occupancies(results, output_path):
    instances = []
    # MinkowskiSparseTensor to dense tensor
    occupancy = results['frustum']['geometry'].dense(shape=torch.Size([1,1,256,256,256]))[0].squeeze(0).squeeze(0)
    instance_indicator = results['panoptic']['panoptic_instances']
    
    # Extract occupancy grid per instance
    for instance_label in range(instance_indicator.max()+1):
        if instance_label in [0,1,2]:
            # "nothing", "wall" or "floor"
            continue
        
        instance = torch.where(
            instance_indicator == instance_label, 
            torch.tensor(1.0).to(occupancy.device), # TODO: why does occupancy not work?
            torch.tensor(0.0).to(occupancy.device)
        )
        
        instances.append((instance, instance_label))
        
    debug_export(occupancy, output_path)
    
    return instances

def occupancies_to_sdfs(occupancy_instances, output_path):
    """
    need to convert to numpy -> no end-to-end gradient
    """
    sdfs = []
    instance_ids = []
    for occupancy_grid, instance_id in occupancy_instances:
        try:
            vertices, faces, _, _ = measure.marching_cubes(occupancy_grid.clone().detach().cpu().numpy(), level=0)
            
            mesh = trimesh.Trimesh(vertices,faces)
            mesh.apply_transform([  
                [ 1.0,   0.0,  0.0,  0.0],
                [ 0.0,  -1.0,  0.0,  0.0],
                [ 0.0,   0.0, -1.0,  0.0],
                [ 0.0,   0.0,  0.0,  1.0]
            ])
            box = mesh.bounding_box_oriented.bounds
            mesh.export(output_path / f"mesh_panoptic_{instance_id}.obj")
            
            diff = box[1] - box[0]
            diff = np.max(diff)
            center = (box[1] + box[0])/2
            
            x_range = (center[0]-diff/2, center[0]+diff/2)
            y_range = (center[1]-diff/2, center[1]+diff/2)
            z_range = (center[2]-diff/2, center[2]+diff/2)
            
            sdf = SDF(vertices, faces)
            
            resolution = 512
            resulting_volume = generate_volume(x_range, y_range, z_range, resolution)
            sdf_sampled = sdf(resulting_volume.reshape(-1,3))
            sdf_sampled = sdf_sampled.reshape(resolution, resolution, resolution)
            
            vertices, faces, _, _ = measure.marching_cubes(sdf_sampled, level=0)
            sdf_mesh = trimesh.Trimesh(vertices, faces)
            sdf_mesh.export(output_path / f"mesh_sdf_{instance_id}.obj")
            
            sdfs.append(torch.Tensor(sdf_sampled).to(occupancy_grid.device)[None, None, ...])
            instance_ids.append(instance_id)
        except Exception as e:
            print(f"skipping instance {instance_id} {e}")
            
    return sdfs, instance_ids

def instance_ids_to_labels(instance_ids, results):
    id_to_class_mapping = results['panoptic']['panoptic_semantic_mapping']
    class_to_label_mapping = get_class_labels()
    return [class_to_label_mapping[id_to_class_mapping[instance_id]] for instance_id in instance_ids]
    
    

def debug_export(occupancy, output_path):
    try:
        vertices, faces, _, _ = measure.marching_cubes(occupancy.clone().detach().cpu().numpy(), level=0)
        mesh.apply_transform([  
                [ 1.0,   0.0,  0.0,  0.0],
                [ 0.0,  -1.0,  0.0,  0.0],
                [ 0.0,   0.0, -1.0,  0.0],
                [ 0.0,   0.0,  0.0,  1.0]
            ])
        mesh = trimesh.Trimesh(vertices,faces)
        mesh.export(output_path / f"occupancy_panoptic.obj")
    except Exception as e:
        print(f"skipping export of panoptic occupancy grid {e}")

def generate_volume(x_range, y_range, z_range, resolution):
    """
    Generate Coordinate Volume with given coordinate ranges
    """
    # Generate 1D arrays for x, y, and z coordinates
    x_coords = np.linspace(x_range[0], x_range[1], resolution)
    y_coords = np.linspace(y_range[0], y_range[1], resolution)
    z_coords = np.linspace(z_range[0], z_range[1], resolution)

    # Create a 3D grid of coordinates using NumPy's meshgrid
    x, y, z = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')

    # Combine the coordinates into a 3D volume
    volume = np.stack((x, y, z), axis=-1)

    return volume

def sdfusion_clean_image(input_image, input_mask):
    from SDFusion.utils.demo_util import preprocess_image
    import torchvision.transforms as transforms

    input_image = input_image.permute(1,2,0).cpu().numpy()
    #input_mask = input_mask.cpu().numpy()
    input_mask = np.ones(input_image.shape[0:2])

    #img_for_vis, img_clean = preprocess_image(input_image, input_mask)
    mean, std = [0.5, 0.5, 0.5], [0.5, 0.5, 0.5]
    transforms = transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize(mean, std),
        #transforms.Resize((256, 256)),
    ])
    img_clean = transforms(input_image).unsqueeze(0)
    
    return img_clean

def run_sdfusion(instances, output_path):
    # first set up which gpu to use
    import os
    gpu_ids = 0
    os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_ids}"
    
    # import libraries
    import numpy as np
    from IPython.display import Image as ipy_image
    from IPython.display import display
    from termcolor import colored, cprint


    import torch.backends.cudnn as cudnn
    cudnn.benchmark = True
    import torchvision.utils as vutils

    from SDFusion.models.base_model import create_model
    from SDFusion.utils.util_3d import read_sdf, render_sdf, render_mesh, sdf_to_mesh, save_mesh_as_gif
    
    # options for the model. please check `utils/demo_util.py` for more details
    from SDFusion.utils.demo_util import SDFusionMM2ShapeOpt

    seed = 2023
    opt = SDFusionMM2ShapeOpt(gpu_ids=gpu_ids, seed=seed)
    device = opt.device
    
    # initialize SDFusion model
    ckpt_path = 'SDFusion/saved_ckpt/sdfusion-mm2shape.pth'
    opt.init_model_args(ckpt_path=ckpt_path)

    SDFusion = create_model(opt)
    cprint(f'[*] "{SDFusion.name()}" loaded.', 'cyan')
    
    from SDFusion.utils.demo_util import preprocess_image, get_shape_mask, tensor_to_pil
    import torchvision.transforms as transforms

    # mm2shape
    out_dir = 'outputs/panoptic_sdfuion'
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    ddim_steps = 50
    ddim_eta = 0.1
    uc_scale = 5.
    mask_mode = 'bottom'
    
    output_instances = [] # (instance_id, occupancy_grid)
    for instance in instances:
        txt_img_scales = [(1., 1.)] # [(0., 0.), (1., 0.), (0., 1.), (1., 1.)]
        for txt_scale, img_scale in txt_img_scales:
            instance['img'] = sdfusion_clean_image(instance['img'], instance['mask'])
            #SDFusion.inference(instance, ddim_steps=ddim_steps, ddim_eta=ddim_eta, uc_scale=uc_scale)
            SDFusion.mm_inference(instance, mask_mode=mask_mode, ddim_steps=ddim_steps, ddim_eta=ddim_eta, uc_scale=uc_scale, txt_scale=txt_scale, img_scale=img_scale)
            # save the generation results
            sdf_gen = SDFusion.gen_df
            mesh_gen = sdf_to_mesh(sdf_gen)
            mesh_trim = trimesh.Trimesh(mesh_gen.verts_list()[0].detach().cpu().numpy(),mesh_gen.faces_list()[0].detach().cpu().numpy())
            mesh_trim.export(output_path / f"mesh_sdfusion_{instance['id']}.obj")
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", type=str, default="data/front3d-sample/rgb_0007.png")
    parser.add_argument("--output", "-o", type=str, default="output/sample_0007/")
    parser.add_argument("--config-file", "-c", type=str, default="configs/front3d_sample.yaml")
    parser.add_argument("--model", "-m", type=str, default="data/panoptic_front3d_v2.pth")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)

    args = parser.parse_args()

    main(args)