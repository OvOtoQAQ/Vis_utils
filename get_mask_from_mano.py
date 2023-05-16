import cv2
import torch
import numpy as np
import os
import pickle
import math 
import smplx
from tqdm import tqdm
def FOV_to_intrinsics(fov_degrees, device='cpu'):
    """
    Creates a 3x3 camera intrinsics matrix from the camera field of view, specified in degrees.
    Note the intrinsics are returned as normalized by image size, rather than in pixel units.
    Assumes principal point is at image center.
    """

    focal_length = float(1 / (math.tan(fov_degrees * 3.14159 / 360) * 1.414))
    intrinsics = torch.tensor([[focal_length, 0, 0.5], [0, focal_length, 0.5], [0, 0, 1]], device=device)
    return intrinsics

model = smplx.create("smpl_models/mano/MANO_RIGHT.pkl", 'mano')

hand_data_path = 'datasets/dart/eva3d_dart/dart_eva3d_part0_256.pkl'
img_root = 'datasets/dart/eva3d_dart/img_0_256'
img_eg = cv2.imread('datasets/dart/eva3d_dart/img_0_256/0.png',cv2.IMREAD_UNCHANGED)
dst_root = 'datasets/dart/eva3d_dart/seg_0_256_mano'
data = None
with open(hand_data_path, 'rb') as f:
    data = pickle.load(f)
for i in tqdm(range(50000)):
    idx = str(i)
    subject = data[idx]
    R = subject['camera_rotation'][0] # identity
    trans = subject['camera_translation'][0]
    global_orient = subject['global_orient'][0]
    hand_pose = subject['hand_pose'][0]
    verts = subject['vertex'][0][:778,:]
    # print(type(verts))
    # print(verts.shape)
    verts = np.dot(R, verts.T).T + trans
    
    fov_degrees = 30
    data_size = 256.
    K = FOV_to_intrinsics(fov_degrees).numpy()
    resize_intri = np.array([[data_size, 0.,0.],
                            [0., data_size,0.],
                            [0.,0.,1.]])
    K =  resize_intri @ K
    verts_img = np.dot(K, verts.T).T
    verts_img[:, :2] = np.round(verts_img[:, :2] / verts_img[:, 2:3])
    verts_img = verts_img.astype(np.int32)


    mask = np.zeros_like(img_eg)
    for f in model.faces:
        triangle = np.array([[verts_img[f[0]][0], verts_img[f[0]][1]], [verts_img[f[1]][0], verts_img[f[1]][1]], [verts_img[f[2]][0], verts_img[f[2]][1]]])
        cv2.fillConvexPoly(mask, triangle, (255,255,255,255))
    cv2.imwrite(os.path.join(dst_root, idx+'.png'), mask)

# root1 = 'datasets/dart/eva3d_dart/seg_0_256'
# root2 = 'datasets/dart/eva3d_dart/seg_0_256_mano'
# for i in range(50):
#     img1 = cv2.imread(os.path.join(root1, str(i)+'.png'), cv2.IMREAD_UNCHANGED)
#     img2 = cv2.imread(os.path.join(root2, str(i)+'.png'), cv2.IMREAD_UNCHANGED)
#     diff = img1-img2
#     cv2.imwrite('./test/'+str(i)+'.png', diff)
    

