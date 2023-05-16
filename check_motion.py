import torch
import numpy as np
from torchvision import transforms
from copy import deepcopy

def ckmt():
    print("checking motion")
    smpl_v = torch.load('smpl_v.pt')
    real_imgs = torch.load('real_imgs.pt')
    trans = torch.load('trans.pt')
    joints = torch.load('joints.pt')
    actual_vox_bbox = torch.load('actual_vox_bbox_canonical.pt')
    predefinedbbox = torch.load('predefinedbbox.pt')
    root  = torch.load('root.pt')
    local_bbox = torch.load('bbox_local_list.pt')
    # bbox_list = torch.stack([torch.cat((t[0], t[1])).reshape(2,3) for t in actual_vox_bbox]).reshape(-1,3)
    canonical_bbox_list = [torch.cat((t[0], t[1])).reshape(2,3).detach().cpu().numpy() for t in actual_vox_bbox]
    p_bboxlist = [torch.cat((t[0], t[1])).reshape(2,3).detach().cpu().numpy() for t in predefinedbbox]
    local_bbox_list = [torch.cat((t[0], t[1])).reshape(2,3).detach().cpu().numpy() for t in local_bbox]
    # print(local_bbox_list)

    j_rot = torch.load('joints_rotated_list.pt')
    rot_j = torch.stack(j_rot).reshape(-1,3).detach().cpu().numpy()
    # print(rot_j.shape)


    inv_transform = transforms.Compose([
            transforms.Normalize((-1, -1, -1), (2, 2, 2))
        ])
    real_imgs = inv_transform(real_imgs)
    real_imgs = real_imgs.cpu().detach().numpy()[0].transpose(1,2,0)
    smpl_v_np = smpl_v.detach().cpu().numpy()
    trans_np = trans.detach().cpu().numpy()
    joints_np = joints.detach().cpu().numpy()
    root = root.detach().cpu().numpy()


    fov_degrees = 30
    data_size = 256.
    import math
    import matplotlib.pyplot as plt
    def FOV_to_intrinsics(fov_degrees):
        """
        Creates a 3x3 camera intrinsics matrix from the camera field of view, specified in degrees.
        Note the intrinsics are returned as normalized by image size, rather than in pixel units.
        Assumes principal point is at image center.
        """

        focal_length = float(1 / (math.tan(fov_degrees * 3.14159 / 360) * 1.414))
        intrinsics = np.array([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]])
        return intrinsics
    K = FOV_to_intrinsics(fov_degrees)
    resize_intri = np.array([[data_size, 0.,0.],
                            [0., data_size,0.],
                            [0.,0.,1.]])
    K =  resize_intri @ K
    R = np.eye(3)
    proj_v = (K @ ((R @ smpl_v_np.T).T + trans_np).T).T
    proj_joints = (K @ ((R @ joints_np.T).T + trans_np).T).T
    proj_rot_joints = (K @ ((R @ rot_j.T).T + trans_np).T).T

    plt.imshow(real_imgs)
    plt.plot(proj_v[:,0]/proj_v[:,2], proj_v[:,1]/proj_v[:,2], 'go')
    plt.plot(proj_joints[:,0]/proj_joints[:,2], proj_joints[:,1]/proj_joints[:,2], 'ro')
    plt.plot(proj_rot_joints[:,0]/proj_rot_joints[:,2], proj_rot_joints[:,1]/proj_rot_joints[:,2], 'yo')

    ## visualize step by steo
    # x1 = proj_joints[:,0]/proj_joints[:,2]
    # y1 = proj_joints[:,1]/proj_joints[:,2]
    # x2 = proj_rot_joints[:,0]/proj_rot_joints[:,2]
    # y2 = proj_rot_joints[:,1]/proj_rot_joints[:,2]
    # for ss in range(16):
    #     plt.plot(x1[ss], y1[ss], 'ro')
    #     plt.plot(x2[ss], y2[ss], 'yo')
    #     plt.savefig('vis_motion_offline.png')

    # given (xyz_min, xyz_max) of bbox, visualize bbox in img
    
    # for bbox in p_bboxlist: # predefined bbox
    for bbox in local_bbox_list: # deformed bbox
    # for bbox in canonical_bbox_list: # canonical bbox
        # bbox = bbox_list[0]
        xyz_min, xyz_max = bbox
        vertices = np.array([    (xyz_min[0], xyz_min[1], xyz_min[2]),
        (xyz_max[0], xyz_min[1], xyz_min[2]),
        (xyz_max[0], xyz_max[1], xyz_min[2]),
        (xyz_min[0], xyz_max[1], xyz_min[2]),
        (xyz_min[0], xyz_min[1], xyz_max[2]),
        (xyz_max[0], xyz_min[1], xyz_max[2]),
        (xyz_max[0], xyz_max[1], xyz_max[2]),
        (xyz_min[0], xyz_max[1], xyz_max[2])
        ])
        edges = np.array([[0, 1], [1, 2], [2, 3], [3, 0],
                    [4, 5], [5, 6], [6, 7], [7, 4],
                    [0, 4], [1, 5], [2, 6], [3, 7]])
        # edges = np.array([[0, 1]])

        proj_vert = (K @ ((R @ vertices.T).T + trans_np).T).T
        x , y = proj_vert[:,0]/proj_vert[:,2], proj_vert[:,1]/proj_vert[:,2]
        for edge in edges:
            v1idx, v2idx = edge
            vx = [x[v1idx], x[v2idx]]
            vy = [y[v1idx], y[v2idx]]
            plt.plot(vx,vy,color='blue', linewidth=2)
        # plt.savefig('vis_motion_offline.png')



    plt.savefig('vis_motion_offline.png')
    plt.close()
