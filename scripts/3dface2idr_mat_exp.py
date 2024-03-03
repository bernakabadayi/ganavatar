# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


import numpy as np
import os
import torch
import json
import argparse
import scipy.io
import sys

print('os.getcwd()', os.getcwd())
sys.path.append('')
sys.path.append(os.getcwd())
sys.path.append(os.getcwd() + '/Deep3DFaceRecon_pytorch')
from Deep3DFaceRecon_pytorch.models.bfm import ParametricFaceModel

parser = argparse.ArgumentParser()
parser.add_argument('--in_root', type=str, default="", help='process folder')
parser.add_argument('--outdir', type=str, default="", help='output folder')
args = parser.parse_args()
in_root = args.in_root

npys = sorted([x for x in os.listdir(in_root) if x.endswith(".mat")])

mode = 1 
outAll={}

face_model = ParametricFaceModel(bfm_folder='Deep3DFaceRecon_pytorch/BFM')

# os.os.makedirs(expdir, exist_ok=True)
expdir =  args.outdir
for src_filename in npys:
    
    
    print('src_filename', src_filename)
    src = os.path.join(in_root, src_filename)
    
    dict_load = scipy.io.loadmat(src)
    angle = dict_load['angle']
    print(angle)
    trans = dict_load['trans'][0]
    exp = dict_load['exp'][0]
    print('exp', exp)


    out_p = os.path.join(expdir, src_filename.replace(".mat", ".txt"))
    np.savetxt(out_p, exp, delimiter='\n', fmt='%.8f')