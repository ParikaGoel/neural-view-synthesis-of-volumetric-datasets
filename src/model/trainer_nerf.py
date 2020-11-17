import os
import glob
import torch
import numpy as np
from model import models
from model import metrics
from data import data_utils
from data import visualize
import torch.utils.data as torchdata
from model import dataset_loader_nerf as loader
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.configure()
        self.create_model()
        self.create_dataloader()

    def configure(self):
        # Bounding volume parameters
        vol_type = self.cfg.vol_type
        center = torch.Tensor(self.cfg.vol_params[:3]).to(device)
        radius = self.cfg.vol_params[3]
        self.vol_params = (vol_type, center, radius)

        # Setup logging
        logdir = os.path.join(os.getcwd(), "logs")
        self.modeldir = os.path.join(os.getcwd(), "models")
        os.makedirs(logdir, exist_ok=True)
        os.makedirs(self.modeldir, exist_ok=True)
        self.writer = SummaryWriter(logdir)
        # Write out config parameters.
        with open(os.path.join(logdir, "config.yaml"), "w") as f:
            f.write(self.cfg.pretty())
    

    def create_dataloader(self):
        datadir = self.cfg.datadir

        i_train = []
        i_val = []
        for f in sorted(glob.glob(datadir + "/train/*.json")):
            filename = f[f.rfind('_') + 1:f.rfind('.')]
            i_train.append(filename)
        
        for f in sorted(glob.glob(datadir + "/val/*.json")):
            filename = f[f.rfind('_') + 1:f.rfind('.')]
            i_val.append(filename)

        self.dataset_train = loader.DatasetLoad(data_list=i_train, data_dir=os.path.join(datadir, "train"),
                                                        cfg=self.cfg)
        self.dataloader_train = torchdata.DataLoader(self.dataset_train, batch_size=self.cfg.batch_size, 
                                                    shuffle=True, num_workers=2, drop_last=False)
        
        self.dataset_val = loader.DatasetLoad(data_list=i_val, data_dir=os.path.join(datadir, "val"),
                                                        cfg=self.cfg)
        self.dataloader_val = torchdata.DataLoader(self.dataset_val, batch_size=self.cfg.batch_size, 
                                                    shuffle=False, num_workers=2, drop_last=False)


    def create_model(self):
        """
        Create a new model or load model from saved checkpoint
        """
        self.model = models.NeRF(num_layers=self.cfg.num_layers,
                                    hidden_size=self.cfg.hidden_size,
                                    skip_connect=self.cfg.skip_connect,
                                    input_ch= self.cfg.input_ch,
                                    output_ch=self.cfg.output_ch).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                        lr=self.cfg.optimizer.lr,
                                        weight_decay=self.cfg.optimizer.weight_decay)

    def print_grad_vals(self):
        for param in self.model.parameters():
            print(param.grad)
    
    def train(self, epoch):
        self.model.train()
        batch_loss = 0.0
        batch_img_loss = 0.0
        batch_output_vis = []
        batch_diff_vis = []
        batch_target_vis = []

        if epoch == 175:
            print("Epoch 175 reached")
    
        for idx, sample in enumerate(self.dataloader_train):
            pts = sample['points'][0].to(device)
            viewdirs = sample['viewdirs'][0].to(device)
            target_rgb = sample['target'][0].to(device)
            
            assert(pts.shape[0] == target_rgb.shape[0])

            batch_pts = [pts[i : i + self.cfg.n_points_in_batch] for i in range(0, pts.shape[0], self.cfg.n_points_in_batch)]
            batch_viewdirs = [viewdirs[i : i + self.cfg.n_points_in_batch] for i in range(0, pts.shape[0], self.cfg.n_points_in_batch)]
            batch_target = [target_rgb[i : i + self.cfg.n_points_in_batch] for i in range(0, pts.shape[0], self.cfg.n_points_in_batch)]
            
            pred_rgb = []

            for batch_id in range(len(batch_pts)):
                target = batch_target[batch_id]
                mask = torch.ones(size=target.shape,dtype=torch.bool,device=target.device)
                input_pts = batch_pts[batch_id]
                pts_shape = input_pts.shape
                input_pts = torch.reshape(input_pts, [-1, pts_shape[-1]])
                viewdirs = torch.reshape(batch_viewdirs[batch_id],[-1,3])
                
                # Run network
                raw = self.model(input_pts)
                raw = torch.reshape(raw, list(pts_shape[:-1]) + [4])

                # Compute opacities and colors
                sigma_a = torch.nn.functional.relu(raw[...,3])
                rgb = torch.sigmoid(raw[...,:3])

                z_vals = sample['z_vals'][0].to(device)
                # z_vals = z_vals.unsqueeze_(0).expand((pts_shape[0],z_vals.shape[-1]))
                one_e_10 = torch.tensor([1e10], dtype=z_vals.dtype, device=z_vals.device)
                dists = torch.cat(
                    (
                        z_vals[..., 1:] - z_vals[..., :-1],
                        one_e_10.expand(z_vals[..., :1].shape),
                    ),
                    dim=-1,
                )
                alpha = 1.0 - torch.exp(-sigma_a * dists)

                # alpha = 1.0 - torch.exp(-sigma_a)
                
                weights = alpha * data_utils.cumprod_exclusive(1.0 - alpha + 1e-10)

                rgb = (weights[..., None] * rgb).sum(dim=-2)
                
                # compute losses
                loss = torch.nn.functional.mse_loss(rgb, target)

                loss.register_hook(lambda grad: print(grad))

                # backward pass and optimize
                loss.backward()
                # log
                # batch_img_loss += loss.item() * (input_pts.shape[0] / (pts.shape[0] * pts.shape[1]))
                pred_rgb.append(rgb.detach())
                batch_loss = loss
                
            # batch_loss += batch_img_loss/(batch_id+1)
            # batch_loss += batch_img_loss
            self.optimizer.step()
            self.optimizer.zero_grad()

            # visualize images on tensorboard
            if idx in [0,1,2,3,4] and (epoch+1) % self.cfg.save_every == 0:
                pred_rgb = torch.cat(pred_rgb,dim=0)
                res = int(np.sqrt(pred_rgb.shape[0]))
                output = torch.reshape(pred_rgb,[res,res,3]).permute(2,0,1)
                output = torch.clamp(output, min=0.0, max=1.0)
                target_rgb = torch.reshape(target_rgb,[res,res,3]).permute(2,0,1)
                batch_target_vis.append(target_rgb)
                batch_diff_vis.append(torch.abs(target_rgb-output))
                batch_output_vis.append(output)
            
        # log losses
        self.writer.add_scalar('train_loss',batch_loss/(idx+1),epoch+1)

        if (epoch+1) % self.cfg.save_every == 0:
            self.writer.add_images('target', torch.stack(batch_target_vis,dim=0), epoch+1)
            self.writer.add_images('output', torch.stack(batch_output_vis,dim=0), epoch+1)
            self.writer.add_images('diff', torch.stack(batch_diff_vis,dim=0), epoch+1)

    def val(self, epoch):
        self.model.eval()
        batch_l1 = 0.0
        batch_mse = 0.0
        batch_psnr = 0.0
        batch_img_l1 = 0.0
        batch_img_mse = 0.0
        batch_img_psnr = 0.0
        batch_output_vis = []
        batch_diff_vis = []
        batch_target_vis = []
    
        with torch.no_grad():
            for idx, sample in enumerate(self.dataloader_val):
                pts = sample['points'][0].to(device)
                viewdirs = sample['viewdirs'][0].to(device)
                target_rgb = sample['target'][0].to(device)
                # mask = sample['mask'][0].to(device)

                assert(pts.shape[0] == target_rgb.shape[0])

                batch_pts = [pts[i : i + self.cfg.n_points_in_batch] for i in range(0, pts.shape[0], self.cfg.n_points_in_batch)]
                batch_viewdirs = [viewdirs[i : i + self.cfg.n_points_in_batch] for i in range(0, viewdirs.shape[0], self.cfg.n_points_in_batch)]
                batch_target = [target_rgb[i : i + self.cfg.n_points_in_batch] for i in range(0, target_rgb.shape[0], self.cfg.n_points_in_batch)]
                # batch_mask = [mask[i : i + self.cfg.n_points_in_batch] for i in range(0, mask.shape[0], self.cfg.n_points_in_batch)]

                pred_rgb = []

                for batch_id in range(len(batch_pts)):
                    target = batch_target[batch_id]
                    # mask = batch_mask[batch_id].unsqueeze(1).expand(target.shape)
                    mask = torch.ones(size=target.shape,dtype=torch.bool,device=target.device)
                    input_pts = batch_pts[batch_id]
                    if self.cfg.use_viewdirs:
                        input_viewdir = batch_viewdirs[batch_id].unsqueeze(1).expand(input_pts.shape)
                        if self.cfg.model_name == 'NeRF':
                            input_pts = torch.cat([input_pts, input_viewdir],dim=-1)
                        else:
                            input_viewdir = torch.reshape(input_viewdir, [-1, 3])

                    pts_shape = input_pts.shape
                    input_pts = torch.reshape(input_pts, [-1, pts_shape[-1]])
                    # input_pts = data_utils.input_mapping(input_pts, self.input_map, self.cfg.map_points, self.cfg.map_viewdirs, self.cfg.points_type, self.cfg.model_name)

                    # Run network

                    if self.cfg.model_name == 'NeRF':
                        raw = self.model(input_pts)
                    else:
                        raw = self.model(input_pts, input_viewdir)
                    raw = torch.reshape(raw, list(pts_shape[:-1]) + [4])

                    # Compute opacities and colors
                    rgb, sigma_a = raw[...,:3], raw[...,3]
                    sigma_a = torch.nn.functional.relu(sigma_a)
                    rgb = torch.sigmoid(rgb)

                    if self.cfg.n_samples != 1:
                        z_vals = sample['z_vals'][0].to(device)
                        one_e_10 = torch.tensor([1e10], dtype=torch.float32, device=device)
                        dists = torch.cat(
                            (
                                z_vals[..., 1:] - z_vals[..., :-1],
                                one_e_10.expand(z_vals[..., :1].shape),
                            ),
                            dim=-1,
                        )
                        alpha = 1.0 - torch.exp(-sigma_a * dists)
                    else:
                        alpha = 1.0 - torch.exp(-sigma_a)
                    weights = alpha * data_utils.cumprod_exclusive(1.0 - alpha + 1e-10)

                    rgb = (weights[..., None] * rgb).sum(dim=-2)
                    
                    # compute losses
                    loss_mse = metrics.mse(target, rgb, mask)
                    loss_l1 = metrics.l1(target, rgb, mask)
                    loss_psnr = metrics.psnr(target, rgb, mask)

                    # log
                    batch_img_l1 += loss_l1.item() * (input_pts.shape[0] / (pts.shape[0] * pts.shape[1]))
                    batch_img_mse += loss_mse.item() * (input_pts.shape[0] / (pts.shape[0] * pts.shape[1]))
                    batch_img_psnr += loss_psnr.item() * (input_pts.shape[0] / (pts.shape[0] * pts.shape[1]))
                    # batch_img_l1 += loss_l1.item()
                    # batch_img_mse += loss_mse.item()
                    # batch_img_psnr += loss_psnr.item()
                    pred_rgb.append(rgb.detach())
                    
                batch_l1 += batch_img_l1 #/(batch_id+1)
                batch_mse += batch_img_mse #/(batch_id+1)
                batch_psnr += batch_img_psnr #/(batch_id+1)

                # visualize images on tensorboard
                if idx in [0,1,2,3,4] and (epoch+1) % self.cfg.save_every == 0:
                    pred_rgb = torch.cat(pred_rgb,dim=0)
                    res = int(np.sqrt(pred_rgb.shape[0]))
                    output = torch.reshape(pred_rgb,[res,res,3]).permute(2,0,1)
                    output = torch.clamp(output, min=0.0, max=1.0)
                    target_rgb = torch.reshape(target_rgb,[res,res,3]).permute(2,0,1)
                    batch_target_vis.append(target_rgb)
                    batch_diff_vis.append(torch.abs(target_rgb-output))
                    batch_output_vis.append(output)
            
        # log losses
        self.writer.add_scalar('val_mse',batch_mse/(idx+1),epoch+1)
        self.writer.add_scalar('val_l1',batch_l1/(idx+1),epoch+1)
        self.writer.add_scalar('val_psnr',batch_psnr/(idx+1),epoch+1)

        if (epoch+1) % self.cfg.save_every == 0:
            self.writer.add_images('val_target', torch.stack(batch_target_vis,dim=0), epoch+1)
            self.writer.add_images('val_output', torch.stack(batch_output_vis,dim=0), epoch+1)
            self.writer.add_images('val_diff', torch.stack(batch_diff_vis,dim=0), epoch+1)
    

    def start(self):
        print('Start training')

        for epoch in range(0, self.cfg.max_epochs):
            self.train(epoch)
            # self.val(epoch)
            
            # self.scheduler.step()

            # if (epoch+1) % 10 == 0:
            #     torch.save({'epoch': epoch+1, 'input_map':self.input_map, 'state_dict': self.model.state_dict(), 
            #                 'optimizer':self.optimizer.state_dict()},
            #                 os.path.join(self.modeldir, '%02d.pth'%(epoch+1)))