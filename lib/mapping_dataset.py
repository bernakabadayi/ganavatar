import os
from torch.utils.data import Dataset
import json
import torch
from PIL import Image
import numpy as np

class Gan360_Dataset(Dataset):
    def __init__(self, data_dir, transform=None, target_transform=None):
        

        self.data_dir = data_dir
        self.bfmw_path = os.path.join(data_dir, 'bfmw_m.json')
        with open(self.bfmw_path) as json_file:
            self.data = json.load(json_file)
        
        self.data = self.data['labels'][0:900] # update n samples
        self.transform = transform
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        
        sample = self.data[index]
        img_name = 'img' + sample['img']

        img_path = os.path.join(self.data_dir, 'img', img_name)
        assert os.path.exists(img_path), 'Image not found: {}'.format(img_path)
        if os.path.exists(img_path):
            img_arr =  np.asarray(Image.open(img_path))
        
        return img_name, torch.FloatTensor(sample['exp']), torch.FloatTensor(sample['w']), torch.FloatTensor(img_arr)

