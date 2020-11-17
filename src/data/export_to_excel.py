import os
import csv
import glob
import numpy as np
from data import data_utils
from data import camera_utils


def export_viewpoints(csv_file, i_train, i_val):
    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = ['split_type','theta', 'phi']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for filename in i_train:
            fov, origin, lookAt, up, near_clip, far_clip, width, height = camera_utils.get_cam_info(cam_file)
            spherical_origin = np.rad2deg(data_utils.cartesian_to_spherical(origin).numpy()).astype(np.int)
            writer.writerow({'split_type':'train', 'theta': spherical_origin[0], 'phi': spherical_origin[1]})

        for filename in i_val:
            fov, origin, lookAt, up, near_clip, far_clip, width, height = camera_utils.get_cam_info(cam_file)
            spherical_origin = np.rad2deg(data_utils.cartesian_to_spherical(origin).numpy()).astype(np.int)
            writer.writerow({'split_type':'val', 'theta': spherical_origin[0], 'phi': spherical_origin[1]})


def read_viewpoints(csv_file):
    # reading viewpoints from excel file
    val_theta = []
    val_phi = []
    train_theta = []
    train_phi = []

    with open(csv_file, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if row['split_type'] == 'train':
                train_theta.append(row['theta'])
                train_phi.append(row['phi'])
            if row['split_type'] == 'val':
                val_theta.append(row['theta'])
                val_phi.append(row['phi'])
    
    return train_theta, train_phi, val_theta, val_phi