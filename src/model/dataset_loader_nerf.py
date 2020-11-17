import os
import json
import torch
import numpy as np
from PIL import Image
from data import visualize
from data import data_utils
from data import camera_utils

class DatasetLoad(torch.utils.data.Dataset):
    def __init__(self, data_list, data_dir, cfg):
        self.data_list = data_list
        self.data_dir = data_dir
        self.cfg = cfg
        
        # Bounding volume parameters
        vol_type = self.cfg.vol_type
        self.vol_params = ()
        if vol_type == "sphere":
            center = torch.Tensor(self.cfg.vol_params[:3])
            radius = self.cfg.vol_params[3]
            self.vol_params = (vol_type, center, radius)
        else:
            sys.exit("Unsupported bounding volume")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        idx = self.data_list[index]

        gt_file = os.path.join(self.data_dir,"color_" + idx + ".png")
        cam_file = os.path.join(self.data_dir,"camera_" + idx + ".json")
        
        # points, viewdirs, valid_mask, z_vals = data_utils.get_input_samples_on_rays(cam_file, self.vol_params, self.cfg.points_type, self.cfg.viewdir_type, self.cfg.n_samples)

        points, viewdirs, z_vals = data_utils.try_sphere_stepping(cam_file, self.vol_params, self.cfg.points_type, self.cfg.viewdir_type, self.cfg.n_samples)

        gt = Image.open(gt_file)
        gt = np.array(gt).astype(np.float32) / 255. # shape: (H, W, C)
        gt = gt[80:480,80:480,:3]
        gt = torch.reshape(torch.Tensor(gt),[-1,3])

        # for sample in range(points.shape[1]):
        #     visualize.project_input_on_bounding_vol(points[:,sample,:].permute(1,0), gt.permute(1,0), None, os.path.join(self.data_dir,"vis/input_%d_%s.ply"%(sample,idx)))
        
        return {'id':idx, 'points':points, 'viewdirs':viewdirs, 'target':gt, 'z_vals':z_vals}