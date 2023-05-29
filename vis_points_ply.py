########## whole hand at a time, pts are all vertices ################

cur_smpl_v = smpl_v
cur_blend_weights = self.smpl_model.lbs_weights.reshape(-1, self.num_joints)
real_vtx_on_mesh = vtx_on_mesh.reshape(1,-1,3)
per_vtx_inv_transformation = smpl_v_inv.reshape(-1, 4, 4)
cur_vtx_inv_shape_transforms = inv_shape_transforms.reshape(-1, 4, 4)
per_vtx_inv_transformation = torch.matmul(cur_vtx_inv_shape_transforms, per_vtx_inv_transformation)
K = 1 
# idxs = knn_points(real_vtx_on_mesh, cur_smpl_v.reshape(1, -1, 3), K=K)
idxs = torch.arange(0, cur_smpl_v.shape[1]).cuda()
vtx_gather_inv_T = torch.gather(per_vtx_inv_transformation.reshape(1, -1, 1, 4, 4).repeat(1, 1, K, 1, 1), 1, idxs.reshape(1, -1, K, 1, 1).repeat(1, 1, 1, 4, 4)).sum(-3).reshape(1, -1, 4, 4)
vtx_homogen_coord = torch.ones([1, real_vtx_on_mesh.shape[1], 1], dtype=real_vtx_on_mesh.dtype, device=real_vtx_on_mesh.device)
real_vtx_on_mesh_homo = torch.cat([real_vtx_on_mesh, vtx_homogen_coord], dim=2)
rays_vtx_local_on_mesh = torch.matmul(vtx_gather_inv_T, torch.unsqueeze(real_vtx_on_mesh_homo, dim=-1))[:, :, :3, 0]
vtx_c = rays_vtx_local_on_mesh[0].detach().cpu().numpy()
vtx_rgb_c = np.repeat(colors[None,0,:], vtx_c.shape[0], axis=0)
to_save_v = np.concatenate([vtx_c, vtx_rgb_c], axis=-1)
np.savetxt('pts_check_motion/vtx_whole.ply',
            to_save_v,
            fmt='%.6f %.6f %.6f %d %d %d',
            comments='',
            header=(
                'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                vtx_c.shape[0])
            )
########## whole hand at a time, pts are all vertices ################




######### individual parts visulization ############
to_save_vtx_list = []
for i, cur_vox in enumerate(self.vox_list):
    real_vtx_on_mesh_o = vtx_on_mesh[0, vtx_part_idx_list[i],...].reshape(1,-1,3)
    vtx_o = real_vtx_on_mesh_o[0].detach().cpu().numpy()
    rgb_o = np.repeat(colors[None,i,:], vtx_o.shape[0], axis=0)
    to_save_vtx_o = np.concatenate([vtx_o, rgb_o], axis=-1)
    to_save_vtx_list.append(to_save_vtx_o)
    np.savetxt('pts_check_motion/vtx_on_mesh_obs_part_{:d}.ply'.format(i),
            to_save_vtx_o,
            fmt='%.6f %.6f %.6f %d %d %d',
            comments='',
            header=(
                'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                vtx_o.shape[0])
            )
to_save_vtx_o_total = np.concatenate(to_save_vtx_list)
np.savetxt('pts_check_motion/vtx_on_mesh_obs.ply',
            to_save_vtx_o_total,
            fmt='%.6f %.6f %.6f %d %d %d',
            comments='',
            header=(
                'ply\nformat ascii 1.0\nelement vertex {:d}\nproperty float x\nproperty float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header').format(
                to_save_vtx_o_total.shape[0])
            )
######### individual parts visulization ############
