from email.policy import default
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
import json

import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator

# conda activate eg3d_n
# python gen_images_eg3d.py --args=/cfg/datagen/args_wojtek1_neck_3dv.yaml


#----------------------------------------------------------------------------

def parse_range(s: Union[str, List]) -> List[int]:
    '''Parse a comma separated list of numbers or ranges and return a list of ints.

    Example: '1,2,5-10' returns [1, 2, 5, 6, 7]
    '''
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        if m := range_re.match(p):
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=False)
@click.option('--seeds', type=parse_range, help='List of random seeds (e.g., \'0,1,4-6\')', required=True, default='0-2')
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.7, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--outdir', help='Where to save the output images', type=str, required=True, metavar='DIR', default='out')
@click.option('--fov-deg', help='Field of View of camera in degrees', type=int, required=False, metavar='float', default=18.837, show_default=True)
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--args', help='Config file', type=str, required=True, metavar='FILE')
def generate_images(
    network_pkl: str,
    seeds: List[int],
    truncation_psi: float,
    truncation_cutoff: int,
    outdir: str,
    fov_deg: float,
    reload_modules: bool,
    args: str
):

    import yaml
    with open(args) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config)

    network_pkl = config['network']
    print(network_pkl)
    seeds = parse_range(config['seeds'])
    outdir = config['outdir']

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda')
    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    if reload_modules:
        print("Reloading Modules!")
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        #G.rendering_kwargs['ray_start'] = 2.0
        #G.rendering_kwargs['ray_end'] = 3.3
        G = G_new
    
    print(G.rendering_kwargs)
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(outdir + '/w', exist_ok=True)
    os.makedirs(outdir + '/z', exist_ok=True)
    os.makedirs(outdir + '/img', exist_ok=True)

    w_dict = {'labels':[]}
    # Generate images.
    for seed_idx, seed in enumerate(tqdm(seeds)):
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, G.z_dim)).to(device)

        angle_p = 0.0
        angle_y = 0.0

        for angle_y, angle_p in [(angle_y, angle_p)]:
            # Set cam pose
            cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivott', [0, 0, 0]), device=device)
            cam2world_pose = LookAtPoseSampler.sample(np.pi/2 + angle_y, np.pi/2 + angle_p, cam_pivot, radius=config['r'], device=device)
            K = torch.tensor(config['fix_pose'][16:], device=device)
            rt = torch.tensor(config['fix_pose'][:16], device=device)
            camera_params = torch.cat([cam2world_pose.reshape(-1, 16), K.reshape(-1, 9)], 1)
            
            # Generate w and img
            ws = G.mapping(z, camera_params, truncation_psi=truncation_psi, truncation_cutoff=truncation_cutoff)
            img = G.synthesis(ws, camera_params)['image']
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)

            # Save img, z, w
            PIL.Image.fromarray(img[0].cpu().numpy(), 'RGB').save(f'{outdir}/img/img{seed:05d}.png')
            np.save(f'{outdir}/z/img{seed:05d}_z.npy', z.cpu().numpy())
            np.save(f'{outdir}/w/img{seed:05d}_w.npy', ws[0][0].cpu().numpy())
            sample = {'img': f'{seed:05d}.png', 'w': ws[0][0].cpu().numpy().tolist()}
            w_dict['labels'].append(sample)

    with open(os.path.join(outdir,'config.yml'), 'w') as outfile:
        yaml.dump(config, outfile, default_flow_style=False)
    
    with open(os.path.join(outdir,'bfmw.json'), 'w') as outfile:
        json.dump(w_dict, outfile, indent=4)
    
    print('Data generation done. Images saved to ', outdir)

#----------------------------------------------------------------------------

if __name__ == "__main__":
    generate_images() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
