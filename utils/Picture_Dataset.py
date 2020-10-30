# pylint: skip-file
import pandas as pd
import numpy as np
import csv as csv
import torch
from torch.utils.data import Dataset
import cv2
import torchvision.transforms.functional as f

"""
Loading Data from Image Folder and Attach Them With Labels
"""

class PictureDataset(Dataset):
    def __init__(self,file_path,csv_path,idx_column=2,transforms=None):
        self.data_store=pd.read_csv(csv_path)
        self.transforms=transforms
        self.idx_column=idx_column
        self.file_path=file_path
    
    def __len__(self):
        return len(self.data_store)
    
    def __getitem__(self,index):
        if torch.is_tensor(index):
            index=index.tolist()
        image_path=self.file_path+self.data_store.iloc[index,0]
        image=cv2.imread(image_path,0)
        label=self.data_store.iloc[index,self.idx_column]
        image=f.to_pil_image(image)
        sample={'Image':image,'Label':label}
        if self.transforms:
            sample['Image']=self.transforms(sample['Image'])
        return sample

    
