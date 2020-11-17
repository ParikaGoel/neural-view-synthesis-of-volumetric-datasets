import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import glob
import json
import hydra
import torch
import numpy as np
from model import trainer_nerf
from model import trainer_ConvNet
from model import trainer_Conv2Net


def train_model(cfg):
    if cfg.model_name == 'NeRF' or cfg.model_name == 'NeRF2':
        trainer = trainer_nerf.Trainer(cfg)
    elif cfg.model_name == 'Conv2Net':
        trainer = trainer_Conv2Net.Trainer(cfg)
    else:
        trainer = trainer_ConvNet.Trainer(cfg)
    trainer.start()

@hydra.main(config_path="../configs/config.yaml")
def main(cfg):
    # Seed experiment for repeatability
    seed = cfg.randomseed
    np.random.seed(seed)
    torch.manual_seed(seed)

    print("Current Working Directory: ", os.getcwd())
    print("Configuration: ", cfg.feature_mapping, " , ", cfg.mapping_size)
    print("N_samples: ", cfg.n_samples, " , Points in a batch", cfg.n_points_in_batch)

    train_model(cfg)


if __name__ == "__main__":
    main()