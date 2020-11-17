import torch
from data import data_utils
from data import bounding_volume


def test_ray_sphere_intersect():
    ray_origin = torch.tensor([[0.0,0.0,4.0],[0.0,0.0,-4.0],[0.0449, 1.0126, -0.4274]])
    ray_dir = torch.tensor([[0.0, 0.0, -1.0],[0.0, 0.0, -1.0],[0.2853, -0.6552, 0.6995]])
    sphere_center = torch.tensor([0.0,0.0,0.0])
    sphere_radius = 1.0
    # intersection points should be [3, 5] and [inf, inf]
    t = bounding_volume.ray_sphere_intersect(ray_origin, ray_dir, sphere_center, sphere_radius)
    print(t)


def test_ray_cube_intersect():
    ray_origin = torch.tensor([[0.0,0.0,4.0],[0.0,0.0,-4.0]])
    ray_dir = torch.tensor([[0.0, 0.0, -1.0],[0.0, 0.0, -1.0]])
    min_bound = torch.Tensor([-0.5, -0.5, -0.5])
    max_bound = torch.Tensor([0.5, 0.5, 0.5])
    # intersection points should be [3.5, 4.5] and [inf, inf]
    t = bounding_volume.ray_cube_intersect(ray_origin, ray_dir, min_bound, max_bound)
    print(t)


def check_coord_conv():
    origins = torch.tensor([[0.0,0.0,0.0],[0.0,0.0,0.0]])
    points = torch.tensor([[1.0,1.0,1.0],[0.0,0.0,-4.0]])
    rays = points - origins
    rays = rays / torch.norm(rays, dim=1)[:,None]
    angles = data_utils.cartesian_to_spherical(rays)
    print(angles)