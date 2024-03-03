import numpy as np
import os
import torch
import json
import argparse
import scipy.io
import sys
print('Currentdir:', os.getcwd())
sys.path.append('Deep3DFaceRecon_pytorch')

parser = argparse.ArgumentParser()
parser.add_argument('--in_root', type=str, default="", help='process folder')
parser.add_argument('--bfm', type=str, default="", help='bfm folder')
args = parser.parse_args()
in_root = args.in_root


bfmw_path = os.path.join(args.bfm, 'bfmw.json')

with open(bfmw_path) as f:
    bfm_data = json.load(f)

w_bfm = bfm_data['labels']

for w in w_bfm:

    img_name = w['img']
    mat_name = img_name.replace(".png", ".mat")
    src = os.path.join(in_root, 'img' + mat_name)
    dict_load = scipy.io.loadmat(src)
    exp = dict_load['exp'][0]
    w.update({'exp': exp.tolist()})


dst = os.path.join(args.bfm, 'bfmw_m.json')
with open(dst, 'w') as f:
    json.dump(bfm_data, f, indent=4)

