# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""
from typing import Dict, List

import torch
from torch import nn

from lib import config
from ..utils import ModuleResult
from structures import SegmentationMask, FieldList

from .rpn import rpn
from .roi_heads import roi_heads


class GeneralizedRCNN(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        output_channel = in_channels[2]

        self.rpn = rpn.build_rpn(output_channel)

        self.roi_heads = None
        if config.MODEL.INSTANCE2D.ROI_HEADS.USE:
            self.roi_heads = roi_heads.build_roi_heads(output_channel)

        self.matching_overlap_threshold = 0.5

    def forward(self, features: Dict, targets: List[FieldList] = None, is_validate: bool = False) -> ModuleResult:
        if is_validate:
            self.train()
        losses = {}

        bounding_boxes_gt = [target.get_field('bbox2d').to('cuda', non_blocking=True) for target in targets]

        features = features["blocks"][2:3]

        results_detection, losses_rpn = self.rpn(features, bounding_boxes_gt)
        losses.update(losses_rpn)

        if config.MODEL.INSTANCE2D.GT_PROPOSAL:
            results_detection = [target.copy_with_fields([]) for target in bounding_boxes_gt]

            for result_item in results_detection:
                result_item.add_field("objectness", torch.ones(len(result_item.bbox)).cuda())

        if self.roi_heads:
            results_detection, losses_roi = self.roi_heads(features, results_detection, bounding_boxes_gt)
            losses.update(losses_roi)

        # Matching
        if is_validate or self.training:
            score_key = "objectness"
        else:
            score_key = "scores2d"

        boxes, masks, raws, locations = self.match_process(results_detection, bounding_boxes_gt)

        results = {"boxes": [box.bbox for box in boxes],
                   "masks": [mask.get_mask_tensor() for mask in masks],
                   "raw": [raw for raw in raws],
                   "locations": [location - 1 for location in locations],
                   "label": [box.get_field("label") for box in boxes]
                   }

        if boxes[0].has_field(score_key):
            results[f"{score_key}"] = [box.get_field(score_key) for box in boxes]

        for name, loss in losses.items():
            losses[name] = config.MODEL.INSTANCE2D.LOSS_WEIGHT * loss

        return losses, results

    def match_process(self, predictions, targets):
        boxes_matched = []
        instance_matched = []
        raw_matched = []
        instance_locations_matched = []

        for proposals_per_image, targets_per_image in zip(predictions, targets):
            if len(proposals_per_image) > 0:

                matched_proposals = self.match_proposals_to_targets(proposals_per_image, targets_per_image)
                boxes = matched_proposals
                boxes_matched.append(boxes)

                segmentation_masks = SegmentationMask(matched_proposals.get_field("mask2d"), targets_per_image.size, mode="mask")
                instance_matched.append(segmentation_masks)

                raw_matched.append(matched_proposals.get_field("mask2draw"))

                instance_locations = matched_proposals.get_field("instance_locations")
                instance_locations_matched.append(instance_locations)
            else:
                boxes = proposals_per_image[[]]  # empty
                boxes_matched.append(boxes)

                segmentation_masks = targets_per_image.get_field("mask2d")[[]]  # empty
                instance_matched.append(segmentation_masks)

                raw_matched.append(segmentation_masks.get_mask_tensor())

                locations = targets_per_image.get_field("mask2dInstance")[[]]  # empty
                instance_locations_matched.append(locations)

        return boxes_matched, instance_matched, raw_matched, instance_locations_matched

    def match_proposals_to_targets(self, proposals, targets):
        locations = []
        matched_proposal_indices = []
        for target_mask, target_location in zip(targets.get_field("mask2d"), targets.get_field("mask2d_instance")):
            for proposal_index, proposal_mask in enumerate(proposals.get_field("mask2d")):

                if proposal_index not in matched_proposal_indices:
                    overlap = metrics.iou.compute_iou(proposal_mask, target_mask.get_mask_tensor())
                    if overlap > self.matching_overlap_threshold:
                        locations.append(target_location)
                        matched_proposal_indices.append(proposal_index)
                        break

        matched_proposals = proposals[matched_proposal_indices]
        matched_proposals.add_field("instance_locations", torch.tensor(locations, dtype=torch.long))
        return matched_proposals
