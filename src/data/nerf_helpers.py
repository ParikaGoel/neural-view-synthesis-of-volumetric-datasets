import os
import glob
import json
import torch
import numpy as np
from PIL import Image
from data import visualize
from data import data_utils

def transforms_to_poses(transforms_file, out_dir):
    with open(transforms_file, 'r') as fp:
        transforms_data = json.load(fp)
    
    frames = transforms_data['frames']
    for frame in frames:
        filename = frame['file_path']
        filename = filename[filename.rfind('/')+1:]
        
        file = out_dir + filename + '.json'
        cam_data = {}
        cam_data['rotation_angle_x'] = frame['rotation']
        cam_data['pose'] = frame['transform_matrix']
        cam_data['width'] = 400
        cam_data['height'] = 400
        cam_data['fov'] = 40.0
        cam_data['farZ'] = 6.0
        cam_data['nearZ'] = 2.0

        with open(file, 'w') as fp:
            json.dump(cam_data, fp)
    
    print('Finished separating poses for each frame')

def project_img_on_sphere(color_file, cam_file, ply_file):
    center = torch.Tensor([0.0, 0.0, 0.0])
    radius = 3.0
    vol_params = ('sphere', center, radius)

    camera_angle_x = 0.6911112070083618
    near = 2.0
    far = 6.0


    colors, points, viewdirs, valid_mask = get_input_data(color_file, cam_file, vol_params, True, camera_angle_x, near, far, 'cartesian', 'cartesian')
    bg_mask = torch.eq(colors[0,...],0) & torch.eq(colors[1,...],0) & torch.eq(colors[2,...],0)
    valid_mask = valid_mask & ~bg_mask
    visualize.project_input_on_bounding_vol(points, colors, valid_mask, ply_file)


def project_imgs_on_sphere(data_dir):
    for f in sorted(glob.glob(data_dir + "/*.json")):
        timestamp = f[f.rfind('/')+1:f.rfind('.')]
        cam_file = f
        color_file = os.path.join(data_dir, timestamp+".png")
        ply_file = os.path.join(data_dir, timestamp+".ply")
        project_img_on_sphere(color_file, cam_file, ply_file)
    

def read_data_from_disk(img_file, cam_file, half_res, camera_angle_x):
    img = Image.open(img_file)
    if half_res:
        img = img.resize(size=(400, 400), resample=Image.NEAREST)
    
    img = np.array(img).astype(np.float32) / 255.

    H, W = img.shape[:2]
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    # if half_res:
    #     focal = focal / 2

    with open(cam_file, 'r') as fp:
        cam_info = json.load(fp)
        pose = torch.Tensor(cam_info['pose']).float()
    
    img = torch.Tensor(img[...,:3]).permute(2,0,1)
    
    return img, pose, [H, W, focal]


def get_rays(H, W, focal, c2w):
    """
    Get ray origins, directions from a pinhole camera.
    rays_o -> tensor of shape [W, H, 3]
    rays_d -> tensor of shape [W, H, 3]
    """
    j, i = torch.meshgrid(torch.arange(W, dtype=torch.float32),
                       torch.arange(H, dtype=torch.float32))
    dirs = torch.stack([(i-W*.5)/focal, -(j-H*.5)/focal, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_d = rays_d / torch.norm(rays_d,dim=-1)[...,None]
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d


def get_input_data(img_file, cam_file, vol_params, half_res, camera_angle_x, near, far, points_type, viewdir_type):
    img, pose, [H, W, focal] = read_data_from_disk(img_file, cam_file, half_res, camera_angle_x)

    # Get the ray, viewdirs and 3d points
    rays = get_rays(H, W, focal, pose)
    points, viewdirs, valid_mask = data_utils.generate_input(rays, near, far, vol_params, points_type, viewdir_type)
    
    return img, points, viewdirs, valid_mask


def get_input_samples_on_rays(img_file, cam_file, vol_params, half_res, camera_angle_x, near, far, points_type, viewdir_type, n_samples):
    img, pose, [H, W, focal] = read_data_from_disk(img_file, cam_file, half_res, camera_angle_x)

    # Get the ray, viewdirs and 3d points
    rays = get_rays(H, W, focal, pose)

    rays_o, rays_d = rays
    rays_o = rays_o[100:300,100:300,:]
    rays_d = rays_d[100:300,100:300,:]
    img = img[:,100:300,100:300]
    rays = [rays_o, rays_d]
    points, z_vals, valid_mask = data_utils.sample_on_rays(rays, near, far, n_samples, vol_params)
    viewdirs = data_utils.get_viewdirs(rays_d, viewdir_type)

    dirs_shape = viewdirs.shape
    viewdirs = torch.reshape(viewdirs,[-1,dirs_shape[-1]])
    viewdirs[valid_mask == False] = 0.0
    
    return img, points, viewdirs, valid_mask, z_vals
