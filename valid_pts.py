# given bbox_min and bbox_max, and given 3d pts coordinates, return pts in bbox
# notice the sum operation

import torch
torch.manual_seed(20)

a = torch.tensor([-1,-1,-1])
b = torch.tensor([1,1,1])
c = torch.randn((2,4,3))
print(c)
valid = (a<c).sum(-1) + (c<b).sum(-1)
valid = (valid == 6)
print(valid)
valid_c = c[valid]
print(valid_c.shape)
