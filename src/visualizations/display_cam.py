import json
import glob
import math
import numpy as np
from data import fileIO
        

def get_cam_frustrum(fov, origin, look_at, up, near_dist, far_dist):
    view_dir = look_at - origin
    view_dir = view_dir / np.linalg.norm(view_dir)

    up  = up / np.linalg.norm(up)
    right = np.cross(up, view_dir)

    h_near = 2 * np.tan(np.radians(fov) / 2) * near_dist
    h_far = 2 * np.tan(np.radians(fov) / 2) * far_dist

    w_near = h_near
    w_far = h_far

    # coordinates for far plane
    fc = origin + view_dir * far_dist 
    ftl = fc + (up * h_far/2) - (right * w_far/2)
    ftr = fc + (up * h_far/2) + (right * w_far/2)
    fbl = fc - (up * h_far/2) - (right * w_far/2)
    fbr = fc - (up * h_far/2) + (right * w_far/2)

    # coordinates for near plane

    nc = origin + view_dir * near_dist 
    ntl = nc + (up * h_near/2) - (right * w_near/2)
    ntr = nc + (up * h_near/2) + (right * w_near/2)
    nbl = nc - (up * h_near/2) - (right * w_near/2)
    nbr = nc - (up * h_near/2) + (right * w_near/2)
    
    return [ftl, ftr, fbl, fbr], [ntl, ntr, nbl, nbr]


def create_cylinder_mesh(radius, p0, p1, stacks=10, slices=10):

    def compute_length_vec3(vec3):
        return math.sqrt(vec3[0]*vec3[0] + vec3[1]*vec3[1] + vec3[2]*vec3[2])
    
    def rotation(axis, angle):
        rot = np.eye(4)
        c = np.cos(-angle)
        s = np.sin(-angle)
        t = 1.0 - c
        axis /= compute_length_vec3(axis)
        x = axis[0]
        y = axis[1]
        z = axis[2]
        rot[0,0] = 1 + t*(x*x-1)
        rot[0,1] = z*s+t*x*y
        rot[0,2] = -y*s+t*x*z
        rot[1,0] = -z*s+t*x*y
        rot[1,1] = 1+t*(y*y-1)
        rot[1,2] = x*s+t*y*z
        rot[2,0] = y*s+t*x*z
        rot[2,1] = -x*s+t*y*z
        rot[2,2] = 1+t*(z*z-1)
        return rot


    verts = []
    indices = []
    diff = (p1 - p0).astype(np.float32)
    height = compute_length_vec3(diff)
    for i in range(stacks+1):
        for i2 in range(slices):
            theta = i2 * 2.0 * math.pi / slices
            pos = np.array([radius*math.cos(theta), radius*math.sin(theta), height*i/stacks])
            verts.append(pos)
    for i in range(stacks):
        for i2 in range(slices):
            i2p1 = math.fmod(i2 + 1, slices)
            indices.append( np.array([(i + 1)*slices + i2, i*slices + i2, i*slices + i2p1], dtype=np.uint32) )
            indices.append( np.array([(i + 1)*slices + i2, i*slices + i2p1, (i + 1)*slices + i2p1], dtype=np.uint32) )
    transform = np.eye(4)
    va = np.array([0, 0, 1], dtype=np.float32)
    vb = diff
    vb /= compute_length_vec3(vb)
    axis = np.cross(vb, va)
    angle = np.arccos(np.clip(np.dot(va, vb), -1, 1))
    if angle != 0:
        if compute_length_vec3(axis) == 0:
            dotx = va[0]
            if (math.fabs(dotx) != 1.0):
                axis = np.array([1,0,0]) - dotx * va
            else:
                axis = np.array([0,1,0]) - va[1] * va
            axis /= compute_length_vec3(axis)
        transform = rotation(axis, -angle)
    transform[:3,3] += p0
    verts = [np.dot(transform, np.array([v[0], v[1], v[2], 1.0])) for v in verts]
    verts = [np.array([v[0], v[1], v[2]]) / v[3] for v in verts]
        
    return verts, indices


def write_ply_cam(ply_file, origin, coords):
    radius = 0.001
    color = (213, 100, 100)
    vertex_lst = []
    
    for coord in coords:
        verts, _ = create_cylinder_mesh(radius, origin, coord, stacks=30, slices=10)
        for vert in verts:
            vertex_lst.append(tuple(vert) + color)
    
    verts, _ = create_cylinder_mesh(radius, coords[0], coords[1], stacks=30, slices=10)
    for vert in verts:
        vertex_lst.append(tuple(vert) + color)
    
    verts, _ = create_cylinder_mesh(radius, coords[0], coords[2], stacks=30, slices=10)
    for vert in verts:
        vertex_lst.append(tuple(vert) + color)
    
    verts, _ = create_cylinder_mesh(radius, coords[1], coords[3], stacks=30, slices=10)
    for vert in verts:
        vertex_lst.append(tuple(vert) + color)
    
    verts, _ = create_cylinder_mesh(radius, coords[2], coords[3], stacks=30, slices=10)
    for vert in verts:
        vertex_lst.append(tuple(vert) + color)

    fileIO.write_ply(ply_file, vertex_lst, [], [])

    
def generate_display_cam(ply_file, origin):
    look_at = np.array([0.0, 0.0, 0.0])
    up = np.array([0.0, 1.0, 0.0])
    
    fov = 40.0

    near_dist = 0.05
    far_dist = 0.3

    radius = 0.02

    far_coords, near_coords = get_cam_frustrum(fov, origin, look_at, up, near_dist, far_dist)
    write_ply_cam(ply_file, origin, near_coords)


def vis_cam_on_sphere(datadir):
    for cam_file in glob.glob(datadir+"*.json"):
        fname = cam_file[cam_file.rfind('/')+1:cam_file.rfind('.')]
        with open(cam_file, 'r') as fp:
            cam_info = json.load(fp)
        
        origin = cam_info['origin']
        ply_file = '/home/goel/Thesis/Data/new_cam_vis/' + fname + '.ply'

        generate_display_cam(ply_file, origin)

