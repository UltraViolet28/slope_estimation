#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

from crf_seg.Pointnet.Pointnet import PointNet
from crf_seg.crf.crf_dense import compute_pairwise_potentials
from crf_seg.crf.crf_dense import CRF_layer

# Grd net will have two parts: PointNet and CRF layer

class GrdNet(nn.Module):
    def __init__(self, num_classes, num_points):
        super(GrdNet, self).__init__()
        self.num_classes = num_classes
        self.num_points = num_points
        self.pointnet = PointNet(num_classes, num_points)
        self.crf = CRF_layer(num_classes, num_points)
        
    def forward(self, point_cloud):
        # PointNet
        unary_potentials = self.pointnet(point_cloud)
        
        # CRF
        surface_normals = self.pointnet.surface_normals
        pairwise_potentials = compute_pairwise_potentials(point_cloud, surface_normals)
        inferred_segmentation = self.crf(unary_potentials, pairwise_potentials)
        
        return inferred_segmentation