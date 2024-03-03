# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Generate images and shapes using pretrained network pickle."""

import os
import re
from typing import List, Optional, Tuple, Union
import sys
sys.path.append('../eg3d/eg3d')

import click
import dnnlib
import numpy as np
import PIL.Image
import torch
from tqdm import tqdm
import mrcfile
import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator
import yaml
from PIL import Image
import json

# Mapping network training includes
import argparse
import os
from mapping_dataset import Gan360_Dataset
from torch.utils.data import DataLoader
import torch
from mapping_model import Expression2WNetwork
from datetime import datetime
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_bfm(config):
    path = config['path']
    pose_path = config['pose_path']

    bfm_dict = {}
    with open(pose_path, 'r') as f:
        pose_list = json.load(f)['labels']

    pose_dict = {}
    for pose in pose_list:
        pose_dict[pose[0]] = pose[1]

    keybfm_names = os.listdir(path)
    for bfm_name in keybfm_names:
        bfm_p = os.path.join(path, bfm_name)
        bfmexp = np.genfromtxt(bfm_p)
        bfm_dict[bfm_name] = bfmexp
    
    return bfm_dict, pose_dict

    
#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=False)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.7, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=True, show_default=True)
@click.option('--args', help='Arguments for training', type=str, required=True, metavar='FILE', show_default=True)
def test(
    network_pkl: str,
    truncation_psi: float,
    truncation_cutoff: int,
    class_idx: Optional[int],
    reload_modules: bool,
    args: str
):
    """Test mapping network for GAN expression control

    Examples:
        
    env: eg3d_cu python mapping_test_in_the_wild.py --args ../../cfg/maptest/args_test_nf01_neck.yaml
    gen video: ffmpeg: ffmpeg -framerate 25 -pattern_type glob -i '*.png' output.mp4
    """

    with open(args) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config)

    network_pkl = config['network']
    b_size = config['batch_size']
    bfmw_path = config['bfmw']
    output_dir = config['output_dir']
    test_network = config['test_network']
    gt_img_path = config['gt_img_path']

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    with dnnlib.util.open_url(network_pkl) as f:
        print('1. network_pkl', network_pkl)
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
        
    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!", reload_modules)
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new
    
    # Specify camera pose

    # K = torch.tensor(config['fix_pose'][16:], device=device)
    # rt = torch.tensor(config['fix_pose'][:16], device=device)
    # camera_params = torch.cat([rt.reshape(-1, 16), K.reshape(-1, 9)], 1)
    
    # camera_params_fix = camera_params.repeat([b_size, 1])
    # camera_params = camera_params_fix

    # Load test data
    test_data =  Gan360_Dataset(bfmw_path)
    test_dataloader = DataLoader(test_data, batch_size=config['batch_size'], shuffle=False, num_workers=0, drop_last=True) 
    print('Test dataset has {} samples'.format(len(test_dataloader)))
    
    # Model, loss
    model = Expression2WNetwork(z_dim=64, map_hidden_dim = 64, map_output_dim = 512, hidden=1).to(device=device)
    model.load_state_dict(torch.load(test_network))
    model.eval()
    
    #log
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(output_dir, 'flame_test_{}'.format(timestamp))
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(log_path + '/w', exist_ok=True)
    os.makedirs(log_path + '/img', exist_ok=True)
    os.makedirs(log_path + '/gt', exist_ok=True)
    os.makedirs(log_path + '/pred', exist_ok=True)

    with open(os.path.join(log_path, 'options.txt'), 'w') as f:
        f.write("testing opts")
        f.write('\n\n')
        f.write(str(timestamp))
        f.write('\n\n')
        f.write('\n\n')
        f.write('Model' + str(model))
        f.write('\n\n')
        f.write('\n\n')
        f.write(str(config))
        f.write('\n\n')
        f.write('Number of parameters ' + str(count_parameters(model)))

    psnr_list = []
    with torch.no_grad():
        model.train(False)
        bfm_dict, pose_dict = load_bfm(config)

        for name,bfm in bfm_dict.items():
            bfm_mf = torch.tensor(bfm, dtype=torch.float32).to(device)
            bfm_mf = bfm_mf.repeat([b_size, 1])
            pred = model(bfm_mf)
            w_broadcast = pred.unsqueeze(1).repeat([1, 14, 1])

            krt = pose_dict[name.replace('.txt', '.png')]
            krt = torch.tensor(krt, device=device)
            
            camera_params = krt.repeat([b_size, 1])
            img = G.synthesis(w_broadcast, camera_params)['image']
            
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128)
            img = img.clamp(0, 255).to(torch.uint8)
        
            gt_img = os.path.join(gt_img_path, name.replace('.txt', '.png'))
            gt_img =  np.asarray(Image.open(gt_img))
            gt_img = torch.FloatTensor(gt_img).to(device)
            gt_img = gt_img.clamp(0, 255).to(torch.uint8).reshape(1, 512, 512, 3)

            from skimage.metrics import peak_signal_noise_ratio
            psnr = peak_signal_noise_ratio(gt_img[0].cpu().numpy().astype(np.uint8), img[0].cpu().numpy().astype(np.uint8), data_range=255)

            psnr_list.append(psnr)

            # save latent
            w_name= name.replace('.txt', '')
            np.save(f'{log_path}/w/{w_name}_w.npy', w_broadcast[0][0].cpu().numpy())

            # save imgs
            imgs = [gt_img, img]
            imgs = torch.cat(imgs, dim=2)
            PIL.Image.fromarray(imgs[0].cpu().numpy(), 'RGB').save(f'{log_path}/img/{w_name}.png')
            PIL.Image.fromarray(gt_img[0].cpu().numpy(), 'RGB').save(f'{log_path}/gt/{w_name}.png')
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{log_path}/pred/{w_name}.png')

    print('PSNR avg: ', np.mean(psnr_list))
#----------------------------------------------------------------------------

if __name__ == "__main__":
    test() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
