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
        self.define_loss()
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
    
    def define_loss(self):
        if self.cfg.loss == 'l1':
            self.loss_fun = metrics.l1
        elif self.cfg.loss == 'lpips':
            self.loss_fun = metrics.lpips
        elif self.cfg.loss == 'ssim':
            self.loss_fun = metrics.msssim
        elif self.cfg.loss == 'mse':
            self.loss_fun = metrics.mse
        elif self.cfg.loss == 'fft':
            self.loss_fun = metrics.loss_fft
        else:
            self.loss_fun = metrics.mse


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
            # self.input_ch = (self.cfg.mapping_size * 2) 
            # if self.cfg.map_points:
            #     dim_points = self.cfg.mapping_size * 2
            
            # if self.cfg.use_viewdirs and self.cfg.map_viewdirs:
            #     dim_viewdir = 0

    def generate_input_mapping(self):
        inp_size = 0
        if self.cfg.map_points:
            if self.cfg.points_type == 'spherical':
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
        for scale in [1., 2., 4., 8., 10., 32.]:
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
        self.model = models.ConvNet(num_layers=self.cfg.num_layers,
                                    hidden_size=self.cfg.hidden_size,
                                    skip_connect=self.cfg.skip_connect,
                                    input_ch= self.input_ch,
                                    output_ch=self.cfg.output_ch).to(device)
        
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
    
    def train(self, epoch):
        self.model.train()
        batch_loss = 0.0
        batch_clamped_output = []
        batch_diff = []
        batch_target = []
        batch_input_points = []
        batch_input_viewdirs = []

        for idx, sample in enumerate(self.dataloader_train):
            input = sample['input'].to(device)
            mask = sample['input_mask'].to(device)
            target = sample['target'].to(device)

            target = target[:,:3,...]
            mask = mask.unsqueeze(1).expand(target.shape)

            self.optimizer.zero_grad()
            # forward pass
            output = self.model(input)

            # compute losses
            loss_l1 = metrics.l1(target, output, mask)
            loss_ssim = metrics.msssim(target, output, mask)
            loss = loss_l1 + loss_ssim
            # loss = self.loss_fun(target, output, mask)

            # backward pass and optimize
            loss.backward()
            self.optimizer.step()

            # log
            batch_loss += loss.item()

            # visualize images on tensorboard
            if idx in [0, 1] and (epoch+1) % self.cfg.save_every == 0:
                clamped_output = torch.clamp(output.detach(), min=0.0, max=1.0) * mask
                target = target * mask
                batch_target.append(target.cpu())
                batch_diff.append(torch.abs(target-clamped_output).cpu())
                batch_clamped_output.append(clamped_output.cpu())

            # if idx in [0,1] and epoch == 0:
            #     raw_data = sample['raw_data'].cpu()
            #     if self.cfg.use_viewdirs:
            #         batch_input_viewdirs.append(raw_data[:,1,...])
            #         raw_data = raw_data[:,0,...]
            #     batch_input_points.append(raw_data)

            # visualize alpha and rgb distribution for first image on tensorboard
            # if idx == 0:
            #     self.writer.add_histogram("red", output[0,0,...], epoch+1)
            #     self.writer.add_histogram("green", rgb[0,1,...], epoch+1)
            #     self.writer.add_histogram("blue", rgb[0,2,...], epoch+1)
            
        # log losses
        self.writer.add_scalar('rgb_loss',batch_loss/(idx+1),epoch+1)

        # log input and target images only once
        # if epoch == 0:
        #     batch_input_points = torch.cat(batch_input_points)
        #     if batch_input_points.shape[1] == 3:
        #         batch_input_points = visualize.vis_cartesian_as_matplotfig(batch_input_points)
        #     else:
        #         batch_input_points = visualize.vis_spherical_as_matplotfig(batch_input_points)

        #     if self.cfg.use_viewdirs:
        #         batch_input_viewdirs = torch.cat(batch_input_viewdirs)
        #         if batch_input_viewdirs.shape[1] == 3:
        #             batch_input_viewdirs = visualize.vis_cartesian_as_matplotfig(batch_input_viewdirs)
        #         else:
        #             batch_input_viewdirs = visualize.vis_spherical_as_matplotfig(batch_input_viewdirs)
            
        #     self.writer.add_figure('input_points', batch_input_points,epoch+1)
        #     if self.cfg.use_viewdirs:
        #         self.writer.add_figure('input_viewdirs', batch_input_viewdirs,epoch+1)
        
        if (epoch+1) % self.cfg.save_every == 0:
            self.writer.add_images('rgb_target', torch.cat(batch_target), epoch+1)
            self.writer.add_images('rgb_clamped', torch.cat(batch_clamped_output), epoch+1)
            self.writer.add_images('rgb_diff', torch.cat(batch_diff), epoch+1)
        
        return batch_loss/(idx+1)
    

    def val(self, epoch):
        self.model.eval()
        batch_l1 = 0.0
        batch_lpips = 0.0
        batch_psnr = 0.0
        batch_ssim = 0.0
        batch_mse = 0.0
        batch_fft = 0.0
        batch_clamped_output = []
        batch_diff = []
        batch_target = []
        batch_input_points = []
        batch_input_viewdirs = []

        with torch.no_grad():
            for idx, sample in enumerate(self.dataloader_val):
                input = sample['input'].to(device)
                mask = sample['input_mask'].to(device)
                target = sample['target'].to(device)

                target = target[:,:3,...]
                mask = mask.unsqueeze(1).expand(target.shape)

                # forward pass
                output = self.model(input)
                
                # compute losses
                # lpips = metrics.lpips(target, output, mask)
                l1 = metrics.l1(target, output, mask)
                # mse = metrics.mse(target, output, mask)
                psnr = metrics.psnr(target, output, mask)
                ssim = metrics.msssim(target, output, mask)
                # fft = metrics.loss_fft(target, output, mask)

                # log
                batch_l1 += l1.item()
                # batch_mse += mse.item()
                # batch_lpips += lpips.item()
                batch_psnr += psnr.item()
                batch_ssim += ssim.item()
                # batch_fft += fft.item()


                # visualize images on tensorboard
                if idx in [0,1] and (epoch+1) % self.cfg.save_every == 0:
                    clamped_output = torch.clamp(output, min=0.0, max=1.0) * mask
                    target = target * mask
                    batch_diff.append(torch.abs(target-clamped_output).cpu())
                    batch_clamped_output.append(clamped_output.cpu())

                if epoch == 0 and idx in [0,1]:
                    batch_target.append(target.cpu())
                    # raw_data = sample['raw_data'].cpu()
                    # if self.cfg.use_viewdirs:
                    #     batch_input_viewdirs.append(raw_data[:,1,...])
                    #     raw_data = raw_data[:,0,...]
                    # batch_input_points.append(raw_data)

        # log losses
        self.writer.add_scalar('rgb_val_loss',batch_l1/(idx+1),epoch+1)
        # self.writer.add_scalar('rgb_val_mse',batch_mse/(idx+1),epoch+1)
        # self.writer.add_scalar('rgb_val_lpips',batch_lpips/(idx+1),epoch+1)
        self.writer.add_scalar('rgb_val_psnr',batch_psnr/(idx+1),epoch+1)
        self.writer.add_scalar('rgb_val_ssim',batch_ssim/(idx+1),epoch+1)
        # self.writer.add_scalar('val_fft',batch_fft/(idx+1),epoch+1)

        # log input and target images only once
        if epoch == 0:
            # batch_input_points = torch.cat(batch_input_points)
            # if batch_input_points.shape[1] == 3:
            #     batch_input_points = visualize.vis_cartesian_as_matplotfig(batch_input_points)
            # else:
            #     batch_input_points = visualize.vis_spherical_as_matplotfig(batch_input_points)

            # if self.cfg.use_viewdirs:
            #     batch_input_viewdirs = torch.cat(batch_input_viewdirs)
            #     if batch_input_viewdirs.shape[1] == 3:
            #         batch_input_viewdirs = visualize.vis_cartesian_as_matplotfig(batch_input_viewdirs)
            #     else:
            #         batch_input_viewdirs = visualize.vis_spherical_as_matplotfig(batch_input_viewdirs)
            
            # self.writer.add_figure('test_input_points', batch_input_points,epoch+1)
            # if self.cfg.use_viewdirs:
            #     self.writer.add_figure('test_input_viewdirs', batch_input_viewdirs,epoch+1)
            self.writer.add_images('rgb_val_target', torch.cat(batch_target), epoch+1)
        
        if (epoch+1) % self.cfg.save_every == 0:
            self.writer.add_images('rgb_val_clamped', torch.cat(batch_clamped_output), epoch+1)
            self.writer.add_images('rgb_val_diff', torch.cat(batch_diff), epoch+1)
        
        return batch_mse/(idx+1)

    def start(self):
        print('Start training')

        self.writer.add_text('Summary', self.cfg.text_summary, 1)
        least_val_loss = 50.0
        train_loss_at_best_epoch = 0.0
        best_epoch = 0
        best_model = {}
        for epoch in range(self.start_epoch, self.cfg.max_epochs):
            train_loss = self.train(epoch)
            val_loss = self.val(epoch)

            # if val_loss <= least_val_loss:
            #     least_val_loss = val_loss
            #     train_loss_at_best_epoch = train_loss
            #     best_epoch = epoch+1
            #     best_model['model_state'] = self.model.state_dict()
            #     best_model['optimizer_state'] = self.optimizer.state_dict()
            self.scheduler.step()

            if (epoch+1) % 20 == 0:
                torch.save({'epoch': epoch+1, 'input_map':self.input_map, 'state_dict': self.model.state_dict(), 
                            'optimizer':self.optimizer.state_dict()},
                            os.path.join(self.modeldir, '%02d.pth'%(epoch+1)))
        
        # torch.save({'epoch': best_epoch, 'input_map':self.input_map, 'state_dict': best_model['model_state'], 
        #             'optimizer':best_model['optimizer_state']},
        #             os.path.join(self.modeldir, 'best_model_%02d.pth'%(best_epoch)))

        # print('Train loss of best model: ', train_loss_at_best_epoch)
        # print('Val loss of best model: ', least_val_loss)
        # print('Best epoch: ', best_epoch)
