##########################################################################################
# NOTE: In depth_anything_v2/dinov2.py, change the line 203 mode=bicubic to mode=bilinear
# Otherwise this the following error will occur: NotImplementedError: The operator 
# 'aten::upsample_bicubic2d.out' is not currently implemented for the MPS device. If you
#  want this op to be added in priority during the prototype phase of this feature,
#  please comment on https://github.com/pytorch/pytorch/issues/77764. As a temporary 
# fix, you can set the environment variable `PYTORCH_ENABLE_MPS_FALLBACK=1` to use the 
# CPU as a fallback for this op. WARNING: this will be slower than running natively on MPS.
##########################################################################################

import cv2
import torch
import os

from depth_anything_v2.dpt import DepthAnythingV2

# Enable CPU fallback for unsupported MPS operations
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

raw_img = cv2.imread('assets/examples_MIT/b8.jpg')
#raw_img_tensor = torch.from_numpy(raw_img).float().permute(2, 0, 1).unsqueeze(0).to(DEVICE)  # Convert to tensor and move to device#
depth = model.infer_image(raw_img) # HxW raw depth map in numpy

cv2.imwrite('results/depth_image.png', depth)