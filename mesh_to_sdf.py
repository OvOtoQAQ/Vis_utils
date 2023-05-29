import os
# os.environ['PYOPENGL_PLATFORM'] = 'osmesa'
# os.environ['DISPLAY'] = ':0.0'

import torch
import numpy as np
# from smpl_utils import init_smpl
from smplx import build_layer
from mesh_to_sdf import mesh_to_voxels, mesh_to_sdf
import trimesh
import skimage

#################################### get sdf from tempalte model #############################################

# smpl_cfgs = {
#     'model_folder': '/mnt/lustre/fzhong/clip/smplx/models',
#     'model_type': 'smpl',
#     'gender': 'neutral',
#     'num_betas': 10
# }

# smpl_cfgs = {
#             'model_folder': 'MANO_RIGHT.pkl',
#             'model_type': 'mano',
#             'num_betas': 10
#         }

# def init_mano(model_folder, model_type, num_betas, gender='NEUTRAL', device='cuda'):
#     if device == 'cuda':
#         smpl_model = build_layer(
#             model_folder, model_type = model_type,
#             num_betas = num_betas
#         ).cuda()
#     elif device == 'cpu':
#         smpl_model = build_layer(
#             model_folder, model_type = model_type,#the model_type is mano for DART dataset
#             num_betas = num_betas
#         )
#     return smpl_model

# smpl_model = init_mano(
#     model_folder = smpl_cfgs['model_folder'],
#     model_type = smpl_cfgs['model_type'],
#     num_betas = smpl_cfgs['num_betas']
# )

# vertices = smpl_model.v_template.cpu().numpy().reshape(-1, 3)
# faces = smpl_model.faces.reshape(-1, 3)
# mesh = trimesh.Trimesh(vertices=vertices, faces=faces)

#################################### get sdf from tempalte model #############################################



################################### get sdf from any given mesh ###############################################



mesh = trimesh.load('smpl_template.obj')

size = 1.3
# size = 0.325 # for hand

resolution = 128 
x, y, z = torch.meshgrid(torch.linspace(-size, size, resolution),
                        torch.linspace(-size, size, resolution),
                        torch.linspace(-size, size, resolution))
pts = torch.stack([x, y, z], -1).view(-1, 3).cpu().numpy()

# voxels = mesh_to_voxels(mesh, 128, pad=True)
voxels = mesh_to_sdf(mesh, pts, sign_method='depth').reshape(resolution, resolution, resolution)
# voxels = np.zeros((128,128,128))
# print('start loop')
# for i in range(128):
#     for j in range(128):
#         for k in range(128):
#             voxels[i, j, k] = voxels_sup[i*4, j*4, k*4]
# print('end loop')

with open('test\smpl_template_cloesd_sdf.npy', 'wb') as f:
    np.save(f, voxels)


# vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0)
# mc_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)

# mc_mesh.export('test/smpl_mesh_from_template_sdf.obj')


def extract_mesh_with_marching_cubes(sdf, level_set=0):
    # b, h, w, d, _ = sdf.shape

    # change coordinate order from (y,x,z) to (x,y,z)
    # sdf_vol = sdf[0,...,0].permute(1,0,2).cpu().numpy()


    w, h, d = sdf.shape
    sdf_vol = sdf

    # scale vertices
    verts, faces, _, _ = skimage.measure.marching_cubes(sdf_vol, level_set, mask=sdf_vol!=10086)
    verts[:,0] = (verts[:,0]/float(w)-0.5)
    verts[:,1] = (verts[:,1]/float(h)-0.5)
    verts[:,2] = (verts[:,2]/float(d)-0.5)

    # # fix normal direction
    verts[:,2] *= -1; verts[:,1] *= -1
    mesh = trimesh.Trimesh(verts, faces)

    return mesh, verts, faces

marching_cubes_mesh, _, _ = extract_mesh_with_marching_cubes(voxels, level_set=0)
marching_cubes_mesh = trimesh.smoothing.filter_humphrey(marching_cubes_mesh, beta=0.2, iterations=5)
marching_cubes_mesh.export('test\smpl_mesh_from_template_sdf.obj')
