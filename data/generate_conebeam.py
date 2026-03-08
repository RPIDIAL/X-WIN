import os.path

import torch
from PIL import Image
import numpy as np

import diffdrr
from diffdrr.drr import DRR
from diffdrr.data import load_example_ct
from diffdrr.visualization import plot_drr

volume_path = '/fast/yangz16/NLST/proj_dataset/step2/nii/100206_T0_1.2.840.113654.2.55.311588773585696213046701923278910973620.nii'

nii_name = os.path.basename(volume_path).rstrip('.nii')
subject2 = diffdrr.data.read(
    volume=volume_path,
    orientation='AP',
)
print(subject2)

# Initialize the DRR module for generating synthetic X-rays
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# yaw, pitch, roll
angles_degree = [30, 45, 45]

angles = np.deg2rad(angles_degree).astype(np.float32)

drr = DRR(
    subject2,     # An object storing the CT volume, origin, and voxel spacing
    sdd=1020.0,  # Source-to-detector distance (i.e., focal length)
    height=512,  # Image height (if width is not provided, the generated DRR is square)
    delx=1.0,    # Pixel spacing (in mm).
).to(device)

# Set the camera pose with rotations (yaw, pitch, roll) and translations (x, y, z)
rotations = torch.tensor([angles], device=device)
translations = torch.tensor([[0.0, 700.0, 0.0]], device=device)

# 📸 Also note that DiffDRR can take many representations of SO(3) 📸
# For example, quaternions, rotation matrix, axis-angle, etc...
img = drr(rotations, translations, parameterization="euler_angles", convention="ZXY")

# Plot
img = img[0, 0].cpu().numpy()
img = (img - img.min()) / (img.max() - img.min()) * 255.
img = img.astype(np.uint8)
img = Image.fromarray(img, mode='L')
img.save(f'/fast/yangz16/outputs/x-win/sample_yaw_roll_pitch/conebeam_proj_yaw_pitch_roll_{angles_degree}.png')
