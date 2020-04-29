from torch.utils.data import Dataset
import numpy as np
import io
from PIL import Image
import os

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

class McDataset(Dataset):
    def __init__(self, input_dir, transform=None):
        self.transform = transform
        self.dir_A = input_dir
        
        imgs= os.listdir(self.dir_A)
        self.A_paths = []
        for img in imgs :
            imgpath=os.path.join(self.dir_A,img)
            self.A_paths.append(imgpath)
        self.num = len(self.A_paths)
        self.A_size = len(self.A_paths)
        #print("read meta done")
        self.initialized = False
 
    def __len__(self):
        return self.num
 
    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        A = load_img(A_path)
        if self.transform is not None:
            A = self.transform(A)
        
        return {'A': A, 'path': A_path }