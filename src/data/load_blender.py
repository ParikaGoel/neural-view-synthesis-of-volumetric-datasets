import os
import PIL
import json
import torch
import numpy as np
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def trans_t(t):
    return torch.tensor([[1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [0, 0, 1, t],
                         [0, 0, 0, 1]], dtype=torch.float32).to(device)


def rot_phi(phi):
    phi = torch.tensor([phi], dtype=torch.float32)
    return torch.tensor([[1, 0, 0, 0],
                         [0, torch.cos(phi), -torch.sin(phi), 0],
                         [0, torch.sin(phi), torch.cos(phi), 0],
                         [0, 0, 0, 1]], dtype=torch.float32).to(device)


def rot_theta(th):
    th = torch.tensor([th], dtype=torch.float32)
    return torch.tensor([[torch.cos(th), 0, -torch.sin(th), 0],
                         [0, 1, 0, 0],
                         [torch.sin(th), 0, torch.cos(th), 0],
                         [0, 0, 0, 1]], dtype=torch.float32).to(device)


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.tensor([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=torch.float32).to(device) @ c2w
    return c2w


def load_blender_data(basedir, half_res=True):
    splits = ['train', 'val', 'test']
    metas = {}
    counts = [0]
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)
        counts.append(counts[-1] + len(metas[s]['frames']))

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = []
    poses = []

    save_poses = {}

    for s in splits:
        meta = metas[s]

        for frame in meta['frames']:
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            img = Image.open(fname)
            if half_res:
                # ToDo: check for area interpolation
                img = img.resize(size=(400, 400), resample=PIL.Image.NEAREST)
            imgs.append(np.array(img))
            poses.append(np.array(frame['transform_matrix']))
    
    imgs = np.array(imgs).astype(np.float32) / 255.  # keep all 4 channels (RGBA)
    poses = np.array(poses).astype(np.float32)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)
    if half_res:
        focal = focal / 2.

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]],
                               0).to(device)

    imgs = torch.tensor(imgs).to(device)
    poses = torch.tensor(poses).to(device)

    return imgs, poses, render_poses, [H, W, focal], i_split
