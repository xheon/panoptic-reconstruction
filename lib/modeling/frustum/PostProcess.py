from typing import List, Dict

import torch
from torch import nn

from lib.modeling.utils import ModuleResult


# TODO: assumes batch size = 0

class PostProcess(nn.Module):
    def __init__(self, things_classes: List[int] = None, stuff_classes: List[int] = None) -> None:
        super().__init__()

        self.thing_classes = things_classes
        if things_classes is None:
            self.thing_classes = []

        self.stuff_classes = stuff_classes
        if stuff_classes is None:
            self.stuff_classes = []

    def forward(self, instance_data: Dict[str, torch.Tensor], frustum_data: Dict[str, torch.Tensor]) -> ModuleResult:
        # filter 3d instances by 2d instances
        instances_filtered = filter_instances(instance_data, frustum_data["instance3d"])

        # merge output
        panoptic_instances = torch.zeros_like(frustum_data["geometry"].F)
        panoptic_semantics = {}

        things_start_index = 2  # map wall and floor to id 1 and 2

        surface_mask = frustum_data["geometry"].F.abs() < 1.0

        # Merge things classes
        for index, instance_id in enumerate(instances_filtered.unique()):
            # Ignore freespace
            if instance_id != 0:
                # Compute 3d instance surface mask
                instance_mask: torch.Tensor = frustum_data["instance3d"].F == instance_id
                instance_surface_mask = instance_mask & surface_mask
                panoptic_instance_id = index + things_start_index
                panoptic_instances[instance_surface_mask] = panoptic_instance_id

                # get semantic prediction
                semantic_region = torch.masked_select(frustum_data["semantic3d_label"].F, instance_surface_mask)
                unique_semantic_labels, semantic_counts = torch.unique(semantic_region, return_counts=True)

                # TODO: ignore semantic freespace label?
                max_count, max_count_index = torch.max(semantic_counts, dim=0)
                selected_label = unique_semantic_labels[max_count_index]

                panoptic_semantics[panoptic_instance_id] = selected_label

        # Merge stuff classes
        # Merge floor class
        wall_class = 10
        wall_id = 1
        wall_surface_mask = frustum_data["semantic3d_label"].F == wall_class & surface_mask
        panoptic_instances[wall_surface_mask] = wall_id
        panoptic_semantics[wall_id] = wall_class

        # Merge floor class
        floor_class = 11
        floor_id = 2
        floor_surface_mask = frustum_data["semantic3d_label"].F == floor_class & surface_mask
        panoptic_instances[floor_surface_mask] = floor_id
        panoptic_semantics[floor_id] = floor_class

        # Search label for unassigned surface voxels
        unassigned_voxels = surface_mask & (panoptic_instances == 0).bool()

        panoptic_instances_copy = panoptic_instances.clone()
        for voxel in unassigned_voxels:
            label = nn_search(panoptic_instances_copy, voxel)

            panoptic_instances[0, 0, voxel[0], voxel[1], voxel[2]] = label

        result = {"panoptic_instances": panoptic_instances, "panoptic_semantics": panoptic_semantics}

        return {}, result


def filter_instances(instances2d,  instances3d):
    instances_filtered = torch.zeros_like(instances3d.F)
    instance_ids_2d = (instances2d["locations"][0] + 1)
    for instance_id in instance_ids_2d:
        if instance_id != 0:
            instance_mask = instances3d.F == instance_id
            instances_filtered[instance_mask] = instance_id

    return instances_filtered


def nn_search(grid, point, radius=3):
    start = -radius
    end = radius

    for x in range(start, end):
        for y in range(start, end):
            for z in range(start, end):
                offset = torch.tensor([x, y, z])
                point_offset = point + offset
                label = grid[0, 0, point_offset[0], point_offset[1], point_offset[2]]

                if label != 0:
                    return label

    return 0
