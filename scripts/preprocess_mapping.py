import os
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--indir', type=str, required=True)
args = parser.parse_args()

# run mtcnn needed for conda
command = "python ../eg3d/dataset_preprocessing/ffhq/batch_mtcnn.py --in_root " + args.indir
print(command)
os.system(command)

# run Deep3DFaceRecon
os.chdir('../eg3d/dataset_preprocessing/ffhq/Deep3DFaceRecon_pytorch')
command = "python test.py --img_folder=" + args.indir + " --gpu_ids=0 --name=pretrained --epoch=20"
print(command)
os.system(command)
os.chdir('../')

out_folder = args.indir.split("/")[-2] if args.indir.endswith("/") else args.indir.split("/")[-1]

# add expression to latent code json
bfm_folder = os.path.abspath(os.path.join(args.indir, os.pardir))


print('Currentdir : inside ', os.getcwd())
 
command = "python ../../../scripts/preprocess_bfmw.py --in_root Deep3DFaceRecon_pytorch/checkpoints/pretrained/results/" + out_folder + "/epoch_20_000000 --bfm " + bfm_folder
print(command)
os.system(command)