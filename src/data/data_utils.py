import os
import sys
import json
import torch
import numpy as np
from PIL import Image
from data import camera_utils
from data.bounding_volume import *


def cumprod_exclusive(tensor: torch.Tensor) -> torch.Tensor:
    r"""Mimick functionality of tf.math.cumprod(..., exclusive=True), as it isn't available in PyTorch.

    Args:
    tensor (torch.Tensor): Tensor whose cumprod (cumulative product, see `torch.cumprod`) along dim=-1
      is to be computed.

    Returns:
    cumprod (torch.Tensor): cumprod of Tensor along dim=-1, mimiciking the functionality of
      tf.math.cumprod(..., exclusive=True) (see `tf.math.cumprod` for details).
    """
    # TESTED
    # Only works for the last dimension (dim=-1)
    dim = -1
    # Compute regular cumprod first (this is equivalent to `tf.math.cumprod(..., exclusive=False)`).
    cumprod = torch.cumprod(tensor, dim)
    # "Roll" the elements along dimension 'dim' by 1 element.
    cumprod = torch.roll(cumprod, 1, dim)
    # Replace the first element by "1" as this is what tf.cumprod(..., exclusive=True) does.
    cumprod[..., 0] = 1.0

    return cumprod
    
def cartesian_to_spherical(coords):
    """
    Converts the cartesian coordinates into spherical coordinates
    r = sqrt(x*x+y*y+z*z)
    theta -> angle with y-axis - range from (-pi to pi) -> elevation
    phi -> angle between x and z axis - range from (0 to 2pi) -> azimuth
    Note : This assumes that ray origin is (0,0,0). Translate the ray coordinates before calling the API
    """
    r = torch.norm(coords, dim=-1)
    theta_rad = torch.asin(coords[...,1] / r)
    phi_rad = torch.atan2(coords[...,2] , coords[...,0])
    coords = torch.stack((theta_rad, phi_rad), dim=-1)
    return coords


def spherical_to_cartesian(center, r, theta, phi):
    """
    Converts the spherical to cartesian coordinates
    theta and phi are the angles in radian
    """
    y = r * torch.sin(theta)
    x = r * torch.cos(theta) * torch.cos(phi)
    z = r * torch.cos(theta) * torch.sin(phi)

    coords = torch.stack([x, y, z], dim=-1) + center[None, None, :]

    return coords

# Using this one for eject dataset
# shape of ray_dirs: (H, W, 3)
def get_ray_dirs(height, width, inv_view_proj, cam_origin):
    j, i = torch.meshgrid(torch.arange(height, dtype=torch.float32),
                       torch.arange(width, dtype=torch.float32))

    # Coordinates in screen space (pixel coordinates to NDC)
    screen_pos = torch.stack([((i+0.5) / width) * 2 - 1, ((j+0.5) / height) * 2 - 1, torch.full_like(i,0.9), torch.ones_like(i)], axis=-1)
    # Coordinates in world space
    world_pos = torch.matmul(screen_pos, inv_view_proj)

    world_pos = world_pos[...,:3] / world_pos[...,3,None]
    
    # Ray directions
    ray_dirs = world_pos - cam_origin
    ray_dirs = ray_dirs / torch.norm(ray_dirs,dim=-1)[...,None]
    ray_origs = cam_origin.expand(ray_dirs.shape)
    return ray_origs, ray_dirs

    
def get_3d_points(rays, near=2.0, far=6.0, vol_params=None):
    """
    Get 3d points from ray information
    These 3d points are fed as input into the network
    rays -> tuple (tensor of shape [W, H, 3], tensor of shape [W, H, 3])
    points_3d -> tensor of shape [W, H, 3]
    """
    # ray origin and ray direction
    rays_o, rays_d = rays
    H, W, _ = rays_o.shape
    rays_o = torch.reshape(rays_o, [-1, 3])
    rays_d = torch.reshape(rays_d, [-1, 3])

    # Unit length direction vectors for each ray
    norms = torch.norm(rays_d, dim=1)
    rays_d = rays_d / norms[:, None]

    # Points in space to evaluate model at
    intersect_dist = ray_vol_intersect(rays_o, rays_d, vol_params)

    intersect_dist = torch.min(intersect_dist, dim=1).values

    # Get the 3d points from the intersection distances
    # Take care that intersection distances could be infinity if corresponding ray
    # does not intersect or intersects outside the camera's viewing plane
    mask = ~torch.isinf(intersect_dist)
    intersect_dist = intersect_dist.unsqueeze_(1).expand(intersect_dist.shape[0],3)
    points_3d = torch.empty_like(rays_o).fill_(float("Inf"))
    points_3d[mask] = rays_o[mask] + intersect_dist[mask] * rays_d[mask]

    # We dont care about the intersection points outside the near and far clipping planes
    mask = torch.lt(points_3d[:,2], near) & torch.gt(points_3d[:,2], far)
    points_3d[mask] = float("Inf")
    points_3d = torch.reshape(points_3d,[H, W, -1])

    return points_3d


def sample_on_rays(rays, near=2.0, far=6.0, n_samples=10, vol_params=None):
    # ray origin and ray direction
    rays_o, rays_d = rays
    H, W, _ = rays_o.shape
    rays_o = torch.reshape(rays_o, [-1, 3])
    rays_d = torch.reshape(rays_d, [-1, 3])

    # Unit length direction vectors for each ray
    norms = torch.norm(rays_d, dim=1)
    rays_d = rays_d / norms[:, None]

    z_vals = torch.linspace(near, far, n_samples).to(rays_o)

    points = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,:,None]
    mask = torch.ones(points.shape[0], dtype=torch.bool)

    return points, z_vals, mask


def get_viewdirs(rays_d, type="cartesian"):
    """
    input
    rays -> tuple (tensor of shape [H, W, 3], tensor of shape [H, W, 3])

    returns viewdirs
    if in cartesian coordinate system
    shape: [n_rays, 3]
    each row contains [x, y, z]

    if in spherical coordinate system
    shape: [n_rays, 2]
    each row contains [theta, phi]
    theta -> angle with +y
    phi -> angle betwen x and z
    """
    if type == 'spherical':
        H, W, _ = rays_d.shape 
        rays_d = torch.reshape(rays_d, [-1,3])
        rays_d = cartesian_to_spherical(rays_d)
        rays_d = torch.reshape(rays_d,[H, W, -1])
    
    return rays_d


def generate_input(rays, near_clip, far_clip, vol_params, points_type, viewdir_type):
    rays_o, rays_d = rays
    points_3d = get_3d_points(rays, near_clip, far_clip, vol_params)
    viewdirs = get_viewdirs(rays_d, viewdir_type)
    valid_mask = ~(torch.isinf(points_3d[:,:,0]) | torch.isinf(points_3d[:,:,1]) | torch.isinf(points_3d[:,:,2]))
    points_3d[valid_mask == False] = 0.0
    viewdirs[valid_mask == False] = 0.0

    points = points_3d.permute(2, 0, 1)
    if points_type == 'spherical': 
        height, width = points.shape[1:]
        spherical_points = torch.full(size=(height, width, 2), fill_value=0.0, dtype=torch.float32)
        spherical_points[valid_mask] = cartesian_to_spherical(points_3d[valid_mask])
        points = spherical_points.permute(2, 0, 1)

    viewdirs = viewdirs.permute(2, 0, 1)

    return points, viewdirs, valid_mask


# generate input data
def get_input_data(cam_file, vol_params, points_type, viewdir_type):
    fov, origin, lookAt, up, near_clip, far_clip, width, height = camera_utils.get_cam_info(cam_file)
    near_clip = 0.01
    far_clip = 10.0
    _, inv_view_proj = camera_utils.compute_matrices(fov, width, height, near_clip, far_clip, origin, lookAt, up)
    rays = get_ray_dirs(height, width, inv_view_proj, origin)

    points, viewdirs, valid_mask = generate_input(rays, near_clip, far_clip, vol_params, points_type, viewdir_type)

    return points, viewdirs, valid_mask


def get_input_samples_on_rays(cam_file, vol_params, points_type, viewdir_type, n_samples):
    fov, origin, lookAt, up, near_clip, far_clip, width, height = camera_utils.get_cam_info(cam_file)
    near_clip = 2.0
    far_clip = 6.0
    _, inv_view_proj = camera_utils.compute_matrices(fov, width, height, near_clip, far_clip, origin, lookAt, up)
    rays = get_ray_dirs(height, width, inv_view_proj, origin)

    rays_o, rays_d = rays
    rays_o = rays_o[80:480, 80:480, :]
    rays_d = rays_d[80:480, 80:480, :]
    rays = [rays_o, rays_d]
    points, z_vals, valid_mask = sample_on_rays(rays, near_clip, far_clip, n_samples, vol_params)
    viewdirs = get_viewdirs(rays_d, viewdir_type)

    dirs_shape = viewdirs.shape
    viewdirs = torch.reshape(viewdirs,[-1,dirs_shape[-1]])
    viewdirs[valid_mask == False] = 0.0
    
    return points, viewdirs, valid_mask, z_vals


def try_sphere_stepping(cam_file, vol_params, points_type, viewdir_type, n_samples):
    fov, origin, lookAt, up, near_clip, far_clip, width, height = camera_utils.get_cam_info(cam_file)
    near_clip = 0.01
    far_clip = 4.0
    _, inv_view_proj = camera_utils.compute_matrices(fov, width, height, near_clip, far_clip, origin, lookAt, up)
    rays_o, rays_d = get_ray_dirs(height, width, inv_view_proj, origin)

    rays_o = rays_o[80:480, 80:480, :]
    rays_d = rays_d[80:480, 80:480, :]

    rays_o = torch.reshape(rays_o, [-1, 3])
    rays_d = torch.reshape(rays_d, [-1, 3])

    # Unit length direction vectors for each ray
    norms = torch.norm(rays_d, dim=1)
    rays_d = rays_d / norms[:, None]

    # Points in space to evaluate model at
    intersect_dist = ray_vol_intersect(rays_o, rays_d, vol_params)
    if n_samples == 1:
        intersect_dist = torch.min(intersect_dist, dim=1).values
        mask = ~torch.isinf(intersect_dist)
        points = torch.empty_like(rays_o).fill_(0.0)
        points[mask] = rays_o[mask] + intersect_dist[mask,None] * rays_d[mask]
        points.unsqueeze_(1)
        z_vals = None
    else:
        min_dist = torch.min(torch.max(intersect_dist, dim=0).values)
        max_dist = torch.max(torch.min(intersect_dist, dim=0).values)

        z_vals = torch.linspace(min_dist, max_dist, n_samples).to(rays_o)

        points = rays_o[...,None,:] + rays_d[...,None,:] * z_vals[...,None]
        
    rays_d = torch.reshape(rays_d,[-1,rays_d.shape[-1]])
    
    return points, rays_d, z_vals



def input_mapping(input, input_map, map_points, map_viewdirs, points_type, model_type):
        if input_map is None:
            return input
        else:
            if model_type == 'NeRF':
                mapped_input = torch.matmul(input, torch.transpose(input_map,0,1))
                mapped_input = torch.cat([torch.sin(mapped_input), torch.cos(mapped_input)], dim=1)
            else:
                if map_points and map_viewdirs:
                    mapped_input = torch.transpose(torch.matmul(torch.transpose(input,0,2), torch.transpose(input_map,0,1)),0,2)
                    mapped_input = torch.cat([torch.sin(mapped_input), torch.cos(mapped_input)], dim=0)
                elif map_points:
                    if points_type == 'spherical':
                        points = input[:2]
                        dirs = input[2:]
                    else:
                        points = input[:3]
                        dirs = input[3:]
                    mapped_input = torch.transpose(torch.matmul(torch.transpose(points,0,2), torch.transpose(input_map,0,1)),0,2)
                    mapped_input = torch.cat([torch.sin(mapped_input), torch.cos(mapped_input), dirs], dim=0)

            return mapped_input