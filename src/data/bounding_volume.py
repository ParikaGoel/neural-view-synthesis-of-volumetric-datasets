import sys
import torch

def solve_quadratic(a, b, c):
    """
    API to calculate the roots of quadratic equation ax^2 + bx + c = 0
    Returns the value of roots or None if no roots exist
    """
    discr = b * b - 4 * a * c
    x = torch.empty((discr.shape[0],2),dtype=torch.float32).fill_(float("Inf"))

    mask = torch.eq(discr, 0)
    if mask.any():
        x[mask, 0] = - 0.5 * b[mask] / a[mask]
        x[mask, 1] = - 0.5 * b[mask] / a[mask]
    
    mask = torch.gt(discr, 0)
    if mask.any():
        x[mask, 0] = 0.5 * a[mask] * (-b[mask] + torch.sqrt(discr[mask]))
        x[mask, 1] = 0.5 * a[mask] * (-b[mask] - torch.sqrt(discr[mask]))

    # Negative value represents that the intersection point is behind the ray origin
    # Since we dont render objects behind the camera, we can set negative values to infinity
    x[torch.lt(x, 0)] = float("Inf")

    return x


def ray_sphere_intersect(ray_origin, ray_dir, sphere_center, sphere_radius):
    L = ray_origin - sphere_center
    a = torch.sum(torch.mul(ray_dir, ray_dir),dim=1)
    b = 2 * torch.sum(torch.mul(L, ray_dir),dim=1)
    c = torch.sum(torch.mul(L,L), dim=1) - (sphere_radius * sphere_radius)

    intersect_dist = solve_quadratic(a, b, c)

    return intersect_dist


def ray_cube_intersect(ray_origin, ray_dir, min_bound, max_bound):
    t_min_bound = (min_bound - ray_origin) / ray_dir
    t_max_bound = (max_bound - ray_origin) / ray_dir

    t_min_bound[torch.isinf(t_min_bound)] = float("Inf")
    t_max_bound[torch.isinf(t_max_bound)] = float("Inf")

    tmin = torch.min(t_min_bound, t_max_bound)
    tmin[torch.isinf(tmin)] = -float("Inf")

    t_min_bound[torch.isinf(t_min_bound)] = -float("Inf")
    t_max_bound[torch.isinf(t_max_bound)] = -float("Inf")

    tmax = torch.max(t_min_bound, t_max_bound)
    tmax[torch.isinf(tmax)] = float("Inf")

    intersect_dist = torch.empty((ray_origin.shape[0],2),dtype=torch.float32).fill_(float("Inf"))
    intersect_dist[:,0] = torch.max(tmin, dim=1).values
    intersect_dist[:,1] = torch.min(tmax, dim=1).values

    mask = torch.le(tmin, intersect_dist[:,1][:,None])
    mask = mask[:,0] & mask[:,1] & mask[:,2]
    intersect_dist[~mask] = float("Inf")
    # Negative value represents that the intersection point is behind the ray origin
    # Since we dont render objects behind the camera, we can set negative values to infinity
    intersect_dist[torch.lt(intersect_dist, 0)] = float("Inf")
    
    return intersect_dist


def ray_vol_intersect(ray_origin, ray_dir, vol_params=None):
    """
    Checks whether a ray intersects with the bounding volume or not and
    returns the intersection distances

    Params : 
    ray_origin -> 3D coordinates of ray origins (shape: [n_rays, 3])
    ray_dir -> Unit length direction vectors corresponding to each ray (shape: [n_rays, 3])
    vol_params -> Tuple defining bounding volume (first entry in tuple tells the type: sphere/cube)
    In case of sphere, vol_params -> ("sphere", center, radius)
    In case of cube, vol_params -> ("cube", min_bound, max_bound)

    Returns : intersection distances (shape: [n_rays, 2]).
    Each row contains intersection distance t0 and t1 corresponding to each ray
    t0 and t1 are the distance of the intersection points from the ray origin.
    t0 and t1 are None if ray does not intersect with the sphere
    If ray intersects at only one point, t0 and t1 will be same.
    t0 and t1 can be positive and negative. Negative value means that the intersection 
    point is in direction opposite to ray direction.

    Intersection point can be calculated as = ray_origin + dist * ray_dir
    """
    if vol_params is None:
        # If information about bounding volume is not provided, assume a unit sphere centered at origin
        vol_type = "sphere"
        origin = torch.Tensor([0.0, 0.0, 0.0])
        radius = 1.0
        vol_params = (vol_type, origin, radius)
    
    vol_type = vol_params[0]

    if vol_type == "sphere":
        return ray_sphere_intersect(ray_origin, ray_dir, vol_params[1], vol_params[2])
    elif vol_type == "cube":
        return ray_cube_intersect(ray_origin, ray_dir, vol_params[1], vol_params[2])
    else:
        sys.exit('Unknown bounding volume type. Please check bounding_volume.py')
    

