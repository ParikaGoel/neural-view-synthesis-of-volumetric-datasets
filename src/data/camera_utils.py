import json
import torch
import numpy as np

def get_proj_mat(fov, width, height, zNear, zFar):
    """
    Calculates the projection matrix
    fov : fov in radians
    zNear & zFar: Near & Far clipping plane
    width & height: camera resolution
    """
    alpha = torch.Tensor([0.5 * fov])
    h = torch.cos(alpha) / torch.sin(alpha)
    w = h * height / width
    projMat = torch.zeros((4,4),dtype=torch.float32)
    projMat[0,0] = w
    projMat[1,1] = h
    projMat[2,2] = zFar / (zFar - zNear)
    projMat[2,3] = 1.0
    projMat[3,2] = -(zFar * zNear) / (zFar - zNear)

    return projMat


def get_view_mat(origin, lookAt, up):
    viewMat = torch.eye(4,dtype=torch.float32)
    viewDir = lookAt - origin
    viewDir = viewDir / torch.norm(viewDir)
    right = torch.cross(up, viewDir)
    right = right / torch.norm(right)
    up = torch.cross(viewDir, right)

    viewMat[:3,0] = right
    viewMat[:3,1] = up
    viewMat[:3,2] = viewDir
    viewMat[3,0] = -torch.dot(right, origin)
    viewMat[3,1] = -torch.dot(up, origin)
    viewMat[3,2] = -torch.dot(viewDir, origin)

    return viewMat


def compute_matrices(fov, width, height, zNear, zFar, origin, lookAt, up):
    projMat = get_proj_mat(fov, width, height, zNear, zFar)
    viewMat = get_view_mat(origin, lookAt, up)
    viewProjMat = torch.matmul(viewMat, projMat)
    invViewProjMat = torch.inverse(viewProjMat)
    return viewProjMat, invViewProjMat


def get_cam_info(cam_file):
    with open(cam_file, 'r') as fp:
        cam_info = json.load(fp)
                    
        fov = np.radians(cam_info['fov'], dtype=np.float32)
        origin = torch.Tensor(cam_info['origin'])
        lookAt = torch.Tensor(cam_info['lookAt'])
        up = torch.Tensor(cam_info['up'])
        near_clip = cam_info['nearZ']
        far_clip = cam_info['farZ']
        width = cam_info['width']
        height = cam_info['height']
    
    return fov, origin, lookAt, up, near_clip, far_clip, width, height