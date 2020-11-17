import os
import json
import torch
from data.image_data import ImageData
from data.load_blender import load_blender_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def cache_data(out_dir, cfg):
        images, poses, _, hwf, i_split = load_blender_data(
        cfg.datadir, cfg.half_res)
        print('Loaded blender', images.shape, hwf, cfg.datadir)

        # Cast intrinsics to right types
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]

        # Bounding volume parameters
        vol_type = cfg.vol_type
        vol_params = ()
        if vol_type == "sphere":
            center = torch.Tensor(cfg.vol_params[:3]).to(device)
            radius = cfg.vol_params[3]
            vol_params = (vol_type, center, radius)
        elif vol_type == "cube":
            min_bound = torch.Tensor(cfg.vol_params[:3]).to(device)
            max_bound = torch.Tensor(cfg.vol_params[3:]).to(device)
            vol_params = (vol_type, min_bound, max_bound)
        else:
            sys.exit("Unsupported bounding volume")
        
        i_train, i_val, i_test = i_split
        
        metadata = {
            "split":
            {
            "train": i_train.tolist(),
            "val": i_val.tolist(),
            "test": i_test.tolist()
            },
            "hwf": hwf
        }

        with open(os.path.join(out_dir,'metadata.json'),'w') as fp:
            json.dump(metadata, fp)
        
        save_dir = os.path.join(out_dir,"train")
        os.makedirs(save_dir,exist_ok=True)

        for idx in i_train:
            img = ImageData(idx, images[idx], poses[idx], hwf, cfg)
            img.cache(vol_params, save_dir)
        
        save_dir = os.path.join(out_dir,"val")
        os.makedirs(save_dir,exist_ok=True)

        for idx in i_val:
            img = ImageData(idx, images[idx], poses[idx], hwf, cfg)
            img.cache(vol_params, save_dir)
        
        save_dir = os.path.join(out_dir,"test")
        os.makedirs(save_dir,exist_ok=True)

        for idx in i_test:
            img = ImageData(idx, images[idx], poses[idx], hwf, cfg)
            img.cache(vol_params, save_dir)