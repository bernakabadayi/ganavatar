# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

# This script is to detect the camera poses and expressions for the mapping network
# the difference is it does not include cropping the images at the end.

import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--indir', type=str, required=True)
args = parser.parse_args()

# # run mtcnn needed for Deep3DFaceRecon
# command = "python ../eg3d/dataset_preprocessing/ffhq/batch_mtcnn.py --in_root " + args.indir
# print(command)
# os.system(command)

out_folder = args.indir.split("/")[-2] if args.indir.endswith("/") else args.indir.split("/")[-1]

# # run Deep3DFaceRecon
os.chdir('../eg3d/dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch')
# command = "python test.py --img_folder=" + args.indir + " --gpu_ids=0 --name=pretrained --epoch=20"
# print(command)
# os.system(command)
os.chdir('..')


print('Currentdir : inside ', os.getcwd())

bfmout = args.indir.replace("key", "bfm")
os.makedirs(bfmout, exist_ok=True)

# convert the pose to our format
command = "python ../../../scripts/3dface2idr_mat_exp.py --in_root Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/" + out_folder + "/epoch_20_000000 --out " + bfmout
print(command)
os.system(command)