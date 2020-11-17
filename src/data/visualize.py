import os
import json
import torch
import numpy as np
from PIL import Image
from data import fileIO
from data import data_utils
from data import camera_utils
from matplotlib import pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def project_input_on_bounding_vol(points, colors, mask, ply_file='./vis.ply'):
    if mask is not None:
        points = points[:,mask]
        colors = colors[:,mask]
    
    colors = colors * 255.0
    
    points = torch.transpose(torch.cat((points, colors),0),0,1)

    vertices = []

    for point in points:
        vertices.append(tuple(point))
    
    fileIO.write_ply(ply_file,vertices,[],[])


def project_eject_on_bouding_vol(img_file, cam_file, ply_file='./vis.ply'):
    img = Image.open(img_file)
    width, height = img.size

    center = torch.Tensor([0.0, 0.0, 0.0])
    radius = 1.0
    vol_params = ('sphere', center, radius)

    points, viewdirs, valid_mask = data_utils.get_input_data(cam_file, vol_params, 'cartesian', 'cartesian')

    img = np.array(img).astype(np.float32)
    img = torch.Tensor(img)
    img = img[..., :3]
    img = img.permute(2, 1, 0)

    bg_mask = torch.le(img[0,...],50) & torch.le(img[1,...],50) & torch.le(img[2,...],50)
    mask = valid_mask[0,...] & valid_mask[1,...] & valid_mask[2,...] & ~bg_mask
    points = points[:,mask]
    colors = img[:,mask]
    
    points = torch.transpose(torch.cat((points, colors),0),0,1)
    vertices = []

    for point in points:
        vertices.append(tuple(point))
    
    fileIO.write_ply(ply_file,vertices,[],[])


def project_img_on_bounding_vol(img, focal, c2w, near=2.0, far=6.0, vol_params=None, ply_file='./vis.ply'):
    H, W, _ = img.shape
    
    rays_o, rays_d = data_utils.get_rays(H, W, focal, c2w)
    points = data_utils.get_3d_points([rays_o, rays_d], near, far, vol_params)
    img = torch.reshape(img, [-1, 3])

    valid_points = ~(torch.isinf(points[:,0]) | torch.isinf(points[:,1]) | torch.isinf(points[:,2]))
    valid_points = valid_points & torch.gt(img[:,0], 0.0) & torch.gt(img[:,1], 0.0) & torch.gt(img[:,2],0)

    points = points[valid_points]
    colors = img[valid_points] * 255.0

    points = torch.cat((points, colors),1)
    vertices = []

    for point in points:
        vertices.append(tuple(point))
    
    fileIO.write_ply(ply_file,vertices,[])


def visualize_points(points, colors, ply_file='./vis.ply'):
    if colors is None:
        colors = torch.ones_like(points)
        colors[...,0] = 0
        colors[...,1] = 169
        colors[...,2] = 255
        
    points = torch.cat((points, colors),-1)
    
    vertices = []

    for point in points:
        vertices.append(tuple(point))
    
    fileIO.write_ply(ply_file,vertices,[],[])


def visualize_output(target, output, visdir):
    output = (output.permute(0, 2, 3, 1).cpu().detach().numpy() * 255.0).astype(np.uint8)
    target = (target.permute(0, 2, 3, 1).cpu().numpy() * 255.0).astype(np.uint8)

    n_imgs = output.shape[0]
    for i in range(n_imgs):
        img = Image.fromarray(output[i])
        img.save(os.path.join(visdir, "out%02d.png"%i))
        img = Image.fromarray(target[i])
        img.save(os.path.join(visdir, "gt%02d.png"%i))


def vis_cartesian_as_matplotfig(cartesian_data):
    cartesian_data = (cartesian_data + 1.0) / 2.0
    batch_size = cartesian_data.shape[0]
    fig, axes = plt.subplots(batch_size,3)

    if batch_size > 1:
        for idx in range(batch_size):
            for ch in range(3):
                axes[idx, ch].imshow(cartesian_data[idx, ch, ...])
                axes[idx, ch].set_title("Image %d ch %d "%(idx,ch))
    else:
        for ch in range(3):
            axes[ch].imshow(cartesian_data[0, ch, ...])
    fig.tight_layout(pad=0.5)
    return fig

def vis_spherical_as_matplotfig(spherical_data):
    spherical_data = (spherical_data + np.pi) / (2 * np.pi)
    batch_size = spherical_data.shape[0]
    fig, axes = plt.subplots(batch_size,2)

    if batch_size > 1:
        for idx in range(batch_size):
            for ch in range(2):
                axes[idx, ch].imshow(spherical_data[idx, ch, ...])
    else:
        for ch in range(2):
            axes[ch].imshow(spherical_data[0, ch, ...])

    fig.tight_layout(pad=0.5)
    return fig