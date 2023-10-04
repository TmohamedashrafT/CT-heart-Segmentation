import torch
import glob
from pathlib import Path
import cv2
from torch.utils.data import DataLoader
from utils import aug_imgs
import numpy as np
class Dataset_Generator(Dataset):
  def __init__(self,
               imgs_path,
               masks_paths,
               img_size,
               data_aug=True):
    super(segment_model, self).__init__()
    self.img_files,self.mask_files = self.img_path(imgs_path,masks_paths)
    self.img_size    = img_size
    self.data_aug    = data_aug

  def __len__(self):
    return len(self.img_files)

  def __getitem__(self,index):
    img, mask = self.load_ind(self.img_files[index],self.mask_files[index])
    img,mask  = aug_imgs(img,mask,self.img_size,self.data_aug)
    img       = (img.transpose((2, 0, 1))[::-1]/255.0).astype(np.float32)
    mask      = (mask[None]/255.0).astype(np.float32)
    return torch.tensor(img),torch.round(torch.tensor(mask))

  def load_ind(self,img_path,mask_path):
        img      = cv2.imread(img_path)
        mask     = cv2.imread(mask_path,0)
        if img is None:
          assert f'{img_path} not founded'
        if mask is None:
          assert f'{img_path} not founded'
        return img,mask

  def img_path(self,img_dir,mask_dir):
        img_files,mask_files = [],[]
        path_img,path_mask      = Path(img_dir),Path(mask_dir)
        img_files +=(glob.glob(str(path_img / '**' / '*.*'), recursive=True))
        mask_files+=(glob.glob(str(path_mask / '**' / '*.*'), recursive=True))
        return sorted(img_files),sorted(mask_files)
    
def data_loader(images,
                masks,
                img_size,
                batch_size,
                data_aug,
                shuffle
                ):
  dataset=Dataset_Generator(
          imgs_path    = images,
          masks_paths  = masks,
          img_size     = img_size,
          data_aug     = data_aug,
          )
  loader=DataLoader(
          dataset,
          batch_size=batch_size,
          shuffle=shuffle,
         )
  return loader

