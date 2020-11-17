import os
import glob
import json
import torch
import numpy as np
from PIL import Image
from data import visualize
from data import data_utils
from data import camera_utils


# Visualize backprojection of image


# With this inverse projection matrix, screen pos (200, 200) should map to world position (0.073136 0.106350 0.047128)
# and ray direction (0.034950 -0.881138 0.471567)
def test_ray_dirs():
    invViewProjMat = torch.Tensor([[-0.411944, 0.039855, -0.404342, 0.367583],
                               [-0.000000, -0.161846, -9.113005, 8.284551],
                               [-0.043297, -0.379197, 3.847051, -3.497320],
                               [0.000000, 0.000000, -9.900009, 10.000009]])

    invViewProjMat = torch.transpose(invViewProjMat, 0, 1)

    camOrigin = torch.Tensor([0.040843, 0.920505, -0.388591])
    width = 504
    height = 504

    _, rayDirs = data_utils.get_ray_dirs(width, height, invViewProjMat, camOrigin)
    print(rayDirs[200, 200])



# Below test case should give below results
# View Proj Matrix:
# -2.400988 0.232293 -0.041255 -0.040843
# 0.000000 -0.943309 -0.929803 -0.920505
# -0.252354 -2.210121 0.392516 0.388591
# 0.000000 0.000000 0.909091 1.000000
# Inverse View Proj Matrix:
# -0.411944 -0.000000 -0.043297 0.000000
# 0.039855 -0.161846 -0.379197 0.000000
# -0.404342 -9.113005 3.847051 -9.900009
# 0.367583 8.284551 -3.497320 10.000009
def test_compute_matrices():
    camOrigin = torch.Tensor([0.040843, 0.920505, -0.388591])
    camLookAt = torch.Tensor([0.000000, 0.000000, 0.000000])
    camUp = torch.Tensor([0.000000, -1.000000, 0.000000])
    width = height = 504
    fov = 45.0
    nearClip = 0.1
    farClip = 10.0

    fov = np.radians(fov, dtype=np.float32)

    viewProjMat, invViewProjMat = camera_utils.compute_matrices(fov, width, height, nearClip, farClip, 
                                                                camOrigin, camLookAt, camUp)

    print("View Projection Matrix: ", viewProjMat)
    print("Inverse View Projection Matrix: ", invViewProjMat)


def test_eject_3d():
    img_file = '/home/goel/Thesis/Data/eject/train/Color_20200710-154659.png'
    pose_file = '/home/goel/Thesis/Data/eject/train/camera_20200710-154659.json'

    img = Image.open(img_file)
    width, height = img.size
    
    with open(pose_file, 'r') as fp:
        pose = json.load(fp)
    
    fov = np.radians(pose['fov'])

    viewProjMat, invViewProjMat = camera_utils.compute_matrices(fov, width, height, 0.1, 10.0, 
                                                                np.array(pose['origin']), 
                                                                np.array(pose['lookAt']),
                                                                np.array(pose['up']))
    
    print("View Projection Matrix: ", viewProjMat)
    print("Inverse View Projection Matrix: ", invViewProjMat)

    raysOrigs, rayDirs = data_utils.get_ray_dirs(width, height, invViewProjMat, np.array(pose['origin']))

    print(rayDirs[200, 200])


def project_depth(color_file, depth_file, cam_file):
    colors = Image.open(color_file)
    width, height = colors.size
    colors = torch.Tensor(np.array(colors, dtype=np.uint8))
    colors = torch.reshape(colors[...,:3],(-1,3))

    depth_vals = np.fromfile(depth_file, dtype=np.float32)
    depth_vals = torch.Tensor(np.reshape(depth_vals,(width, height)))

    with open(cam_file, 'r') as fp:
        cam_info = json.load(fp)
    
    fov = np.radians(cam_info['fov'], dtype=np.float32)
    origin = torch.Tensor(cam_info['origin'])
    lookAt = torch.Tensor(cam_info['lookAt'])
    up = torch.Tensor(cam_info['up'])
    near_clip = cam_info['nearZ']
    far_clip = cam_info['farZ']

    _, inv_view_proj_mat = camera_utils.compute_matrices(fov, width, height, near_clip, far_clip, origin, lookAt, up)
    
    j, i = torch.meshgrid(torch.arange(width, dtype=torch.float32),
                       torch.arange(height, dtype=torch.float32))
    # Coordinates in screen space (pixel coordinates to NDC)
    screenPos = torch.stack([((i+0.5) / width) * 2 - 1, ((j+0.5) / height) * 2 - 1, depth_vals, torch.ones_like(i)], axis=-1)
    mask = ~torch.eq(screenPos[...,2],0)
    # Coordinates in world space
    worldPos = torch.matmul(screenPos, inv_view_proj_mat)
    worldPos = worldPos[...,:3] / worldPos[...,3,None]
    worldPos = torch.reshape(worldPos,(-1,3))

    mask = torch.flatten(mask)
    # remove all black colors
    mask = torch.gt(colors[...,0], 50) & torch.gt(colors[...,1], 50) & torch.gt(colors[...,2], 50)

    visualize.visualize_points(points=worldPos[mask], colors=colors[mask], ply_file=depth_file[:depth_file.rfind('.')] + '.ply')


def project_depths(data_dir):
    for f in sorted(glob.glob(data_dir + "/*.bin")):
        timestamp = f[f.rfind('_')+1:f.rfind('.')]
        depth_file = f
        color_file = os.path.join(data_dir, "color_"+timestamp+".png")
        cam_file = os.path.join(data_dir, "camera_"+timestamp+".json")
        project_depth(color_file, depth_file, cam_file)


def project_imgs_on_sphere(data_dir):
    for f in sorted(glob.glob(data_dir + "/*.json")):
        timestamp = f[f.rfind('_')+1:f.rfind('.')]
        cam_file = f
        color_file = os.path.join(data_dir, "color_"+timestamp+".png")
        ply_file = os.path.join(data_dir, "color_"+timestamp+".ply")
        visualize.project_eject_on_bouding_vol(color_file, cam_file, ply_file)