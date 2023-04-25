import vctoolkit as vc


class MANOHand:
  n_keypoints = 21

  n_joints = 21

  center = 4

  root = 0

  labels = [
    'W', #0
    'I0', 'I1', 'I2', #3
    'M0', 'M1', 'M2', #6
    'L0', 'L1', 'L2', #9
    'R0', 'R1', 'R2', #12
    'T0', 'T1', 'T2', #15
    'I3', 'M3', 'L3', 'R3', 'T3' #20, tips are manually added (not in MANO)
  ]

  # finger tips are not keypoints in MANO, we label them on the mesh manually
  mesh_mapping = {16: 333, 17: 444, 18: 672, 19: 555, 20: 744}

  parents = [
    None,
    0, 1, 2,
    0, 4, 5,
    0, 7, 8,
    0, 10, 11,
    0, 13, 14,
    3, 6, 9, 12, 15
  ]

  end_points = [0, 16, 17, 18, 19, 20]

  

vc.joints_to_mesh(xyz, MANOHand.parents, save_path='joints.obj')
