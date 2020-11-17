import os
import sys
import glob
import json
import torch
import numpy as np
from model import models
from model import metrics
from data import visualize
from data import data_utils
import matplotlib.pyplot as plt
from model import dataset_loader
from torchsummary import summary
import torch.utils.data as torchdata
from torch.utils.tensorboard import SummaryWriter


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        self.configure()
        self.cal_input_channels()
        self.generate_input_mapping()
        self.create_model()
        self.create_dataloader()


    def configure(self):
        # Bounding volume parameters
        vol_type = self.cfg.vol_type
        self.vol_params = ()
        if vol_type == "sphere":
            center = torch.Tensor(self.cfg.vol_params[:3]).to(device)
            radius = self.cfg.vol_params[3]
            self.vol_params = (vol_type, center, radius)
        elif vol_type == "cube":
            min_bound = torch.Tensor(self.cfg.vol_params[:3]).to(device)
            max_bound = torch.Tensor(self.cfg.vol_params[3:]).to(device)
            self.vol_params = (vol_type, min_bound, max_bound)
        else:
            sys.exit("Unsupported bounding volume")

        # Setup logging
        logdir = os.path.join(os.getcwd(), "logs")
        self.modeldir = os.path.join(os.getcwd(), "models")
        os.makedirs(logdir, exist_ok=True)
        os.makedirs(self.modeldir, exist_ok=True)
        self.writer = SummaryWriter(logdir)

    
    def cal_input_channels(self):
        dim_points = 3
        dim_viewdir = 0

        if self.cfg.points_type == 'spherical':
            dim_points = 2
        
        if self.cfg.use_viewdirs and self.cfg.viewdir_type == 'cartesian':
            dim_viewdir = 3
        elif self.cfg.use_viewdirs and self.cfg.viewdir_type == 'spherical':
            dim_viewdir = 2
        
        self.input_ch = dim_points + dim_viewdir

        if self.cfg.feature_mapping != 'none':
            self.input_ch = self.input_ch + (self.cfg.mapping_size * 2) 

    def generate_input_mapping(self):
        inp_size = 0
        if self.cfg.map_points and self.cfg.points_type == 'spherical':
            inp_size += 2
        else:
            inp_size += 3
        
        if self.cfg.use_viewdirs and self.cfg.map_viewdirs:
            if self.cfg.viewdir_type == 'cartesian':
                inp_size += 3
            else: 
                inp_size += 2
        
        self.B_dict = {}
        # Standard network - no mapping
        self.B_dict['none'] = None
        # Gaussian Fourier feature mapping
        B_gauss = torch.normal(mean=0, std=1.0, size=(self.cfg.mapping_size, inp_size))
        # Three different scales of Gaussian Fourier feature mappings
        for scale in [1., 10., 32.0]:
            self.B_dict[f'gauss_{scale}'] = B_gauss * scale

        self.input_map = self.B_dict[self.cfg.feature_mapping]
    

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

        self.dataset_train = dataset_loader.DatasetLoad(data_list=i_train, data_dir=os.path.join(datadir, "train"),
                                                        cfg=self.cfg, input_map=self.input_map)
        self.dataloader_train = torchdata.DataLoader(self.dataset_train, batch_size=self.cfg.batch_size, 
                                                    shuffle=True, num_workers=2, drop_last=False)
        
        self.dataset_val = dataset_loader.DatasetLoad(data_list=i_val, data_dir=os.path.join(datadir, "val"),
                                                        cfg=self.cfg, input_map=self.input_map)
        self.dataloader_val = torchdata.DataLoader(self.dataset_val, batch_size=self.cfg.batch_size, 
                                                    shuffle=False, num_workers=2, drop_last=False)

    def create_model(self):
        """
        Create a new model or load model from saved checkpoint
        """
        self.model = models.Conv2Net(hidden_size=self.cfg.hidden_size,
                                input_ch= self.input_ch).to(device)
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), 
                                        lr=self.cfg.optimizer.lr,
                                        weight_decay=self.cfg.optimizer.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size = self.cfg.scheduler.step_decay, gamma = self.cfg.scheduler.lr_decay)
        
        # summary(self.model, (512, 512, 512))

        self.start_epoch = 0

        # load a model from saved checkpoint if provided
        if self.cfg.checkpoint:
            print('loading model: ', self.cfg.checkpoint)
            checkpoint = torch.load(self.cfg.checkpoint, map_location=device)
            self.start_epoch = checkpoint['epoch']
            self.input_map = checkpoint['input_map'].cpu()
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

    def freeze_alpha_layers(self):
        for layers in self.model.layers_nw1:
                for param in layers.parameters():
                    param.requires_grad = False

        for param in self.model.conv_alpha.parameters():
            param.requires_grad = False

    def train(self, epoch):
        self.model.train()
        batch_alpha_loss = 0.0
        batch_rgb_loss = 0.0
        batch_clamped_alpha = []
        batch_blended_rgb  = []
        batch_target_rgb = []
        batch_target_alpha = []

        for idx, sample in enumerate(self.dataloader_train):
            input = sample['input'].to(device)
            mask = sample['input_mask'].to(device)
            target = sample['target'].to(device)

            target_alpha = target[:,3,...].unsqueeze_(1)
            target_rgb = target[:,:3,...]
            rgb_mask = mask.unsqueeze(1).expand(target_rgb.shape)
            alpha_mask = mask.unsqueeze(1)

            self.optimizer.zero_grad()
            # forward pass
            alpha, rgb = self.model(input)

            # compute losses
            alpha_loss = metrics.l1(target_alpha, alpha, alpha_mask)

            clamped_alpha = torch.clamp(alpha, min=0.0, max=1.0)

            blended_rgb = clamped_alpha * rgb

            rgb_loss = metrics.l1(target_rgb, blended_rgb, rgb_mask)

            cb_rgb = torch.clamp(blended_rgb, min=0.0, max=1.0)

            loss = alpha_loss + 10 * rgb_loss

            # backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # log
            batch_alpha_loss += alpha_loss.item()
            batch_rgb_loss += rgb_loss.item()

            # visualize images on tensorboard
            if idx in [0,1,2,3] and (epoch+1) % self.cfg.save_every == 0:
                batch_clamped_alpha.append(clamped_alpha)
                batch_blended_rgb.append(cb_rgb)
                
            if idx in [0,1,2,3] and epoch == 0:
                batch_target_alpha.append(target_alpha)
                batch_target_rgb.append(target_rgb)
                
        # log losses
        self.writer.add_scalar('alpha_loss',batch_alpha_loss/(idx+1),epoch+1)
        self.writer.add_scalar('rgb_loss',batch_rgb_loss/(idx+1),epoch+1)

        # log input and target images only once
        if epoch == 0:
            self.writer.add_images('alpha_target', torch.cat(batch_target_alpha), epoch+1)
            self.writer.add_images('rgb_target', torch.cat(batch_target_rgb), epoch+1)

        if (epoch+1) % self.cfg.save_every == 0:
            self.writer.add_images('alpha_clamped', torch.cat(batch_clamped_alpha), epoch+1)
            self.writer.add_images('rgb_blended', torch.cat(batch_blended_rgb), epoch+1)
   
    def val(self, epoch):
        self.model.eval()
        batch_alpha_loss = 0.0
        batch_rgb_loss = 0.0
        batch_alpha_psnr = 0.0
        batch_rgb_psnr = 0.0
        batch_clamped_alpha = []
        batch_blended_rgb  = []
        batch_target_rgb = []
        batch_target_alpha = []

        with torch.no_grad():
            for idx, sample in enumerate(self.dataloader_val):
                input = sample['input'].to(device)
                mask = sample['input_mask'].to(device)
                target = sample['target'].to(device)

                target_alpha = target[:,3,...].unsqueeze_(1)
                target_rgb = target[:,:3,...]
                rgb_mask = mask.unsqueeze(1).expand(target_rgb.shape)
                alpha_mask = mask.unsqueeze(1)

                # forward pass
                alpha, rgb = self.model(input)

                # compute losses
                alpha_loss = metrics.l1(target_alpha, alpha, alpha_mask)
                alpha_psnr = metrics.psnr(target_alpha, alpha, alpha_mask)

                clamped_alpha = torch.clamp(alpha, min=0.0, max=1.0)

                blended_rgb = clamped_alpha * rgb

                rgb_loss = metrics.l1(target_rgb, blended_rgb, rgb_mask)
                rgb_psnr = metrics.psnr(target_rgb, blended_rgb, rgb_mask)

                cb_rgb = torch.clamp(blended_rgb, min=0.0, max=1.0)

                # log
                batch_alpha_loss += alpha_loss.item()
                batch_rgb_loss += rgb_loss.item()
                batch_alpha_psnr += alpha_psnr.item()
                batch_rgb_psnr += rgb_psnr.item()

                # visualize images on tensorboard
                if idx in [0,1,2,3] and (epoch+1) % self.cfg.save_every == 0:
                    batch_clamped_alpha.append(clamped_alpha)
                    batch_blended_rgb.append(cb_rgb)
                    
                if idx in [0,1,2,3] and epoch == 0:
                    batch_target_alpha.append(target_alpha)
                    batch_target_rgb.append(target_rgb)
                
            # log losses
            self.writer.add_scalar('val_alpha_loss',batch_alpha_loss/(idx+1),epoch+1)
            self.writer.add_scalar('val_rgb_loss',batch_rgb_loss/(idx+1),epoch+1)
            self.writer.add_scalar('val_alpha_psnr',batch_alpha_psnr/(idx+1),epoch+1)
            self.writer.add_scalar('val_rgb_psnr',batch_rgb_psnr/(idx+1),epoch+1)

            # log input and target images only once
            if epoch == 0:
                self.writer.add_images('val_alpha_target', torch.cat(batch_target_alpha), epoch+1)
                self.writer.add_images('val_rgb_target', torch.cat(batch_target_rgb), epoch+1)

            if (epoch+1) % self.cfg.save_every == 0:
                self.writer.add_images('val_alpha_clamped', torch.cat(batch_clamped_alpha), epoch+1)
                self.writer.add_images('val_rgb_blended', torch.cat(batch_blended_rgb), epoch+1)

    def start(self):
        print('Start training')
        
        for epoch in range(self.start_epoch, self.cfg.max_epochs):
            self.train(epoch)
            self.val(epoch)
            self.scheduler.step()

            if (epoch+1) % 5 == 0:
                torch.save({'epoch': epoch+1, 'input_map':self.B_dict[self.cfg.feature_mapping], 'state_dict': self.model.state_dict(), 
                            'optimizer':self.optimizer.state_dict()},
                            os.path.join(self.modeldir, '%02d.pth'%(epoch+1)))
