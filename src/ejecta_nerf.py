# Python file to load the models trained for nerf approach, saving the predicted images 
# and calculating evalutation metrics on predictions

import os
import glob
import json
import torch
import random
import numpy as np
from PIL import Image
from PIL import ImageDraw
from model import models
from model import metrics
from data import camera_utils
from data import data_utils
import matplotlib.pyplot as plt

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# volume params
center = torch.Tensor([0.0, 0.0, 0.0])
radius = 1.0
vol_params = ("sphere", center, radius)

np.random.seed(42)
torch.manual_seed(42)
random.seed(42)

# load saved model
model_name = 'NeRF'
input_ch = 3
points_type = 'cartesian'
viewdir_type = 'cartesian'
n_samples = 128
n_points_in_batch = 5000
out_dir = '/home/goel/Thesis/Code/dvr/outputs/nerf_approach/first_run/lr1e-2/points_128/'
writer_path = os.path.join(out_dir,'logs')
model_path = os.path.join(out_dir,'models/200.pth')

# create the model
if model_name == 'NeRF':
    model = models.NeRF(num_layers=6, hidden_size=128, skip_connect=[4], input_ch = input_ch, output_ch = 4).to(device)
else:
    model = models.NeRF2(num_layers=6, hidden_size=128, skip_connect=[4]).to(device)

checkpoint = torch.load(model_path, map_location=device)
input_map = checkpoint['input_map']
model.load_state_dict(checkpoint['state_dict'])


def run_nerf_on_one_img(cam_file):
    points, viewdirs, _, z_vals = data_utils.try_sphere_stepping(cam_file, vol_params, "cartesian", "cartesian", n_samples)
    points = points.to(device)
    viewdirs = viewdirs.to(device)
    if z_vals is not None:
        z_vals = z_vals.to(device) 
    
    batch_pts = [points[i : i + n_points_in_batch] for i in range(0, points.shape[0], n_points_in_batch)]
    batch_viewdirs = [viewdirs[i : i + n_points_in_batch] for i in range(0, viewdirs.shape[0], n_points_in_batch)]
    
    pred_rgb = []
    for batch_id in range(len(batch_pts)):
        input_pts = batch_pts[batch_id]
        
        if n_samples == 1 and model_name == 'NeRF':
            input_viewdir = batch_viewdirs[batch_id].unsqueeze_(1).expand(input_pts.shape)
            input_pts = torch.cat([input_pts, input_viewdir],dim=-1)
        elif model_name == 'NeRF2':
            input_viewdir = batch_viewdirs[batch_id].unsqueeze_(1).expand(input_pts.shape)
            input_viewdir = torch.reshape(input_viewdir, [-1, 3])
        
        pts_shape = input_pts.shape
        input_pts = torch.reshape(input_pts, [-1, pts_shape[-1]])

        if model_name == 'NeRF':
            raw = model(input_pts)
        elif model_name == 'NeRF2':
            raw = model(input_pts, input_viewdir)
        else:
            print("Check model name")
            exit(0)

        raw = torch.reshape(raw, list(pts_shape[:-1]) + [4])

        # Compute opacities and colors
        rgb, sigma_a = raw[...,:3], raw[...,3]
        sigma_a = torch.nn.functional.relu(sigma_a)
        rgb = torch.sigmoid(rgb)

        if n_samples == 1:
            alpha = 1.0 - torch.exp(-sigma_a)
        else:
            one_e_10 = torch.tensor([1e10], dtype=torch.float32, device=device)
            dists = torch.cat(
                (
                    z_vals[..., 1:] - z_vals[..., :-1],
                    one_e_10.expand(z_vals[..., :1].shape),
                ),
                dim=-1,
            )
            alpha = 1.0 - torch.exp(-sigma_a * dists)

        weights = alpha * data_utils.cumprod_exclusive(1.0 - alpha + 1e-10)
        rgb = (weights[..., None] * rgb).sum(dim=-2)
        pred_rgb.append(rgb.detach())
    
    pred_rgb = torch.cat(pred_rgb,dim=0)
    pred_rgb = torch.reshape(pred_rgb,[400,400,3])
    return pred_rgb


def eval_metrics(cam_file, img_file):
    target_rgb = Image.open(img_file)
    target_rgb = np.array(target_rgb).astype(np.float32) / 255. # shape: (H, W, C)
    target_rgb = torch.Tensor(target_rgb[80:480,80:480,:3]).to(device).permute(2,0,1).unsqueeze_(0)
    mask = torch.ones(size=target_rgb.shape,device=target_rgb.device)

    pred_rgb = run_nerf_on_one_img(cam_file).permute(2,0,1).unsqueeze_(0)

    psnr = metrics.psnr(target_rgb, pred_rgb, mask)
    ssim = metrics.msssim(target_rgb, pred_rgb, mask)

    return psnr, ssim

def save_pred(file):
    pred_rgb = (pred_rgb * 255.0).cpu().numpy().astype(np.uint8)
    pred_rgb = Image.fromarray(pred_rgb)
    pred_rgb.save(file)

def run_nerf(data_dir):
    psnr_val = 0.0
    ssim_val = 0.0
    count = 0

    for cam_file in glob.glob(os.path.join(data_dir,'*.json')):
        id = cam_file[cam_file.rfind('_')+1:cam_file.rfind('.')]
        img_file = os.path.join(data_dir,'color_%s.png'%id)

        psnr,ssim = eval_metrics(cam_file, img_file)

        psnr_val += psnr
        ssim_val += ssim
        count += 1

    psnr_val = psnr_val / count
    ssim_val = ssim_val / count

    return psnr_val, ssim_val

if __name__ == "__main__":
    data_dir = '/home/goel/Thesis/Data/ejecta_27/val'
    psnr_val, ssim_val = run_nerf(data_dir)
    print(psnr_val, ssim_val)
    
