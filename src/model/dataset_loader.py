import os
import json
import torch
import numpy as np
from PIL import Image
from data import visualize
from data import data_utils
from data import nerf_helpers
from data import camera_utils

class DatasetLoad(torch.utils.data.Dataset):
    def __init__(self, data_list, data_dir, cfg, input_map):
        self.data_list = data_list
        self.data_dir = data_dir
        self.input_map = input_map
        self.cfg = cfg
        
        # Bounding volume parameters
        vol_type = self.cfg.vol_type
        self.vol_params = ()
        if vol_type == "sphere":
            center = torch.Tensor(self.cfg.vol_params[:3])
            radius = self.cfg.vol_params[3]
            self.vol_params = (vol_type, center, radius)
        elif vol_type == "cube":
            min_bound = torch.Tensor(self.cfg.vol_params[:3])
            max_bound = torch.Tensor(self.cfg.vol_params[3:])
            self.vol_params = (vol_type, min_bound, max_bound)
        else:
            sys.exit("Unsupported bounding volume")

    def __len__(self):
        return len(self.data_list)

    def input_mapping(self, input):
        if self.input_map is None:
            return input
        else:
            if self.cfg.map_points and self.cfg.map_viewdirs:
                mapped_input = torch.transpose(torch.matmul(torch.transpose(input,0,2), torch.transpose(self.input_map,0,1)),0,2)
                mapped_input = torch.cat([input, torch.sin(mapped_input), torch.cos(mapped_input)], dim=0)
            elif self.cfg.map_points:
                if self.cfg.points_type == 'spherical':
                    points = input[:2]
                    dirs = input[2:]
                else:
                    points = input[:3]
                    dirs = input[3:]
                mapped_input = torch.transpose(torch.matmul(torch.transpose(points,0,2), torch.transpose(self.input_map,0,1)),0,2)
                mapped_input = torch.cat([points, torch.sin(mapped_input), torch.cos(mapped_input)], dim=0)

            return mapped_input


    def __getitem__(self, index):
        idx = self.data_list[index]
        if self.cfg.dataset_name == 'ejecta':
            gt_file = os.path.join(self.data_dir,"color_" + idx + ".png")
            cam_file = os.path.join(self.data_dir,"camera_" + idx + ".json")
        elif self.cfg.dataset_name == 'shapenet':
            gt_file = os.path.join(self.data_dir,idx + ".png")
            cam_file = os.path.join(self.data_dir,idx + ".json")
        elif self.cfg.dataset_name == 'lego' or self.cfg.dataset_name == 'shapenet':
            gt_file = os.path.join(self.data_dir,idx + ".png")
            cam_file = os.path.join(self.data_dir,idx + ".json")
        elif self.cfg.dataset_name == 'ejecta_iso':
            gt_file = os.path.join(self.data_dir,"ao_" + idx + ".bin")
            cam_file = os.path.join(self.data_dir,"camera_" + idx + ".json")
        
        if self.cfg.dataset_name == 'lego':
            gt, points, viewdirs, valid_mask = nerf_helpers.get_input_data(gt_file, cam_file, self.vol_params, self.cfg.half_res, self.cfg.camera_angle_x, self.cfg.near, self.cfg.far, self.cfg.points_type, self.cfg.viewdir_type) 
        else:
            gt = Image.open(gt_file)
            gt = np.array(gt).astype(np.float32) / 255. # shape: (H, W, C)
            gt = torch.Tensor(gt).permute(2,0,1) # shape: (C, H, W)

            points, viewdirs, valid_mask = data_utils.get_input_data(cam_file, self.vol_params, self.cfg.points_type, self.cfg.viewdir_type)
        
        if len(gt.shape) == 2:
            # if we have one channel ground truth, shape would be (H,W)
            # add one dimension for one channel
            gt.unsqueeze_(0)
        
        if self.cfg.use_viewdirs:
            input = torch.cat((points, viewdirs),dim=0)
            raw_data = torch.stack((points, viewdirs),dim=0)
        else:
            input = points
            raw_data = points

        # Project input
        # visualize.project_input_on_bounding_vol(points, gt, valid_mask, os.path.join(os.getcwd(),'./vis_'+idx+'.ply'))

        # Map the input into higher dimensional space
        input = self.input_mapping(input)
        
        return {'id':idx, 'input':input, 'input_mask':valid_mask, 'target':gt, 'raw_data':raw_data}