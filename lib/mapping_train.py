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

import legacy
from camera_utils import LookAtPoseSampler, FOV_to_intrinsics
from torch_utils import misc
from training.triplane import TriPlaneGenerator
import yaml
from torchmetrics import PeakSignalNoiseRatio


# Mapping network training include
import argparse
import os
from mapping_dataset import Gan360_Dataset
from torch.utils.data import DataLoader
import torch
from mapping_model import Expression2WNetwork
from datetime import datetime
import numpy as np
from torch.utils.tensorboard import SummaryWriter


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def logmodel(model):
    print(model)
    print(count_parameters(model))

    
#----------------------------------------------------------------------------

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', required=False)
@click.option('--trunc', 'truncation_psi', type=float, help='Truncation psi', default=0.7, show_default=True)
@click.option('--trunc-cutoff', 'truncation_cutoff', type=int, help='Truncation cutoff', default=14, show_default=True)
@click.option('--class', 'class_idx', type=int, help='Class label (unconditional if not specified)')
@click.option('--reload_modules', help='Overload persistent modules?', type=bool, required=False, metavar='BOOL', default=False, show_default=True)
@click.option('--args', help='Arguments for training', type=str, required=True, metavar='FILE', show_default=True)
def train(
    network_pkl: str,
    truncation_psi: float,
    truncation_cutoff: int,
    class_idx: Optional[int],
    reload_modules: bool,
    args: str
):
    """Train mapping network for GAN expression control

    Example run:
        
    (eg3d) python mapping_train.py --args ../cfg/mapnet/args_train_wojtek1.yaml 
    """
    
    with open(args) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print(config)

    network_pkl = config['network']
    b_size = config['batch_size']
    bfmw_path = config['bfmw']
    output_dir = config['output_dir']
    nsamples = config['nsamples']

    print('Loading networks from "%s"...' % network_pkl)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    with dnnlib.util.open_url(network_pkl) as f:
        G = legacy.load_network_pkl(f)['G_ema'].to(device)
        
    # Specify reload_modules=True if you want code modifications to take effect; otherwise uses pickled code
    # BK: If you change the TriPlaneGenerator code, then it should be true here, o.w. its fine as it is.
    reload_modules = False
    if reload_modules:
        print("Reloading Modules!", reload_modules)
        G_new = TriPlaneGenerator(*G.init_args, **G.init_kwargs).eval().requires_grad_(False).to(device)
        misc.copy_params_and_buffers(G, G_new, require_all=True)
        G_new.neural_rendering_resolution = G.neural_rendering_resolution
        G_new.rendering_kwargs = G.rendering_kwargs
        G = G_new
    
    # specify frontal looking camera
    cam_pivot = torch.tensor(G.rendering_kwargs.get('avg_camera_pivott', [0, 0, 0]), device=device)
    cam2world_pose = LookAtPoseSampler.sample(np.pi/2, np.pi/2, cam_pivot, radius=config['r'], device=device)
    K = torch.tensor(config['fix_pose'][16:], device=device)
    rt = torch.tensor(config['fix_pose'][:16], device=device)
    camera_params = torch.cat([cam2world_pose.reshape(-1, 16), K.reshape(-1, 9)], 1)
    camera_params = camera_params.repeat([b_size, 1])

    # Load training data
    psnr = PeakSignalNoiseRatio().to(device)
    training_data =  Gan360_Dataset(bfmw_path)

    train_dataloader = DataLoader(training_data, batch_size=b_size, shuffle=True, num_workers=0, drop_last=True) 
    print('Training dataset has {} samples'.format(len(training_data)))
    
    val_data =  Gan360_Dataset(bfmw_path) # update dataloader to val set
    val_dataloader = DataLoader(val_data, batch_size=b_size, shuffle=False, num_workers=0, drop_last=True) 
    print('Validation dataset set has {} samples'.format(len(val_data)))

    # Model, opt, loss
    model = Expression2WNetwork(z_dim=64, map_hidden_dim = 64, map_output_dim = 512, hidden=1).to(device=device)
    optimizer = torch.optim.AdamW(lr=config['lr'], weight_decay=0, params=model.parameters(), amsgrad=False)
    loss_fn = torch.nn.L1Loss()
    
    # log
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(output_dir, 'flame_training_{}'.format(timestamp))
    os.makedirs(log_path, exist_ok=True)
    writer = SummaryWriter('runs/training_{}'.format(timestamp))

    with open(os.path.join(log_path, 'options.txt'), 'w') as f:
        f.write("opt for later")
        f.write('\n\n')
        f.write(str(timestamp))
        f.write('\n\n')
        f.write('Training set has instances: ' + str(len(training_data)))
        f.write('\n\n')
        f.write('Val set has instances: ' + str(len(training_data)))
        f.write('\n\n')
        f.write('Model' + str(model))
        f.write('\n\n')
        f.write('Optimizer' + str(optimizer))
        f.write('\n\n')
        f.write(str(config))
        f.write('\n\n')
        f.write('Number of parameters ' + str(count_parameters(model)))
    
    glob_it = 0
    running_loss = 0
    avg_train_psnr = 0.0

    for epoch in tqdm(range(config['epochs'])):
        model.train(True)
        for i, el in enumerate(train_dataloader):
            glob_it += 1
            name, bfm, w , gt_img = el
            bfm, w, gt_img = bfm.to(device), w.to(device), gt_img.to(device)
            
            optimizer.zero_grad()

            # forward
            pred = model(bfm)
            w_broadcast = pred.unsqueeze(1).repeat([1, 14, 1])
            
            # synthesis
            img = G.synthesis(w_broadcast, camera_params)['image'] # img: torch.Size([batch_size, 3, 512, 512])
            img = (img.permute(0, 2, 3, 1) * 127.5 + 128)
            
            # loss
            loss = loss_fn(img, gt_img)
            snr = psnr(img, gt_img)

            # running loss
            avg_train_psnr += snr.item()
            running_loss += loss.item()

            #print("glob_it", glob_it, loss, snr)
            loss.backward()
            optimizer.step()

            if glob_it % config['freq_t'] == 0:
                writer.add_scalars('runningloss', {'train':running_loss/config['freq_t']}, glob_it)
                writer.add_scalars('psnr', {'train':avg_train_psnr/config['freq_t']}, glob_it)

                # tensorboard img
                one_pred_img = img.permute(0, 3, 1, 2)[0].clamp(0, 255).to(torch.uint8)
                one_gt_image = gt_img.permute(0, 3, 1, 2)[0].clamp(0, 255).to(torch.uint8)

                #writer.add_image('GT img', one_gt_image, glob_it) # the dim should be (3, H, W), 
                #writer.add_image('Pred img', one_pred_img, glob_it) 

                c = torch.cat((one_gt_image, one_pred_img), 2)
                writer.add_image('gt vs predicted', c, glob_it) 

                print('Iteration: ', glob_it , '     loss', running_loss/config['freq_t'])

                # flush
                running_loss = 0
                avg_train_psnr = 0.0
                writer.flush()
            
            if glob_it % config['save_freq'] == 0:
                model_path = os.path.join(log_path, 'model_{}_{}.pth'.format(timestamp, glob_it))
                torch.save(model.state_dict(), model_path)
        
        # val
        avg_psnr = 0.0
        if epoch % config['freq_v'] == 0 and epoch > 0:
            with torch.no_grad():
                model.eval()
                running_vloss = 0.0
                for y, vdata in enumerate(val_dataloader): 
                    vname, v_bfm, v_w, v_gt_img = vdata
                    v_bfm, v_w, v_gt_img = v_bfm.to(device), v_w.to(device), v_gt_img.to(device)
                    v_pred = model(v_bfm)
                    v_w_broadcast = v_pred.unsqueeze(1).repeat([1, 14, 1])
                    
                    v_img = G.synthesis(v_w_broadcast, camera_params)['image']
                    v_img = (v_img.permute(0, 2, 3, 1) * 127.5 + 128)
                    v_loss = loss_fn(v_img, v_gt_img)
                    running_vloss += v_loss.item()

                    snr = psnr(v_img, v_gt_img)
                    avg_psnr += snr.item()

                avg_vloss = running_vloss / (y + 1) # val los for 1 step, or iteration
                writer.add_scalars('runningloss', {'val':avg_vloss}, glob_it)
                writer.add_scalars('psnr', {'val':avg_psnr/(y+1)}, glob_it)
            
#----------------------------------------------------------------------------

if __name__ == "__main__":
    train() # pylint: disable=no-value-for-parameter

#----------------------------------------------------------------------------
