import cv2
from glob import glob
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from pydicom import dcmread
import matplotlib.image as mpimg
from utils import get_mask

class MOAIDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform = None, rate = 1.0, valid = True):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = np.array(sorted(glob(image_dir + '/*/*')))
        self.masks = np.array(sorted(glob(mask_dir + '/*/*')))
        self.rate = rate
        self.valid = valid

        if self.valid == True:
            self.images, self.masks = self.images[:self.rate], self.masks[:self.rate]
        else:
            self.images, self.masks = self.images[self.rate:], self.masks[self.rate:]
            
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index, train):
        img_path = self.images[index]  # .dcm으로 저장된 DCM format의 CT 데이터
        mask_path = self.masks[index]  # .png로 저장된 PNG format의 ground truth data
        

        image = dcmread(img_path).pixel_array # shape = (512, 512)
        mask = mpimg.imread(mask_path)
        mask = np.around(mask, 8)

        m_bg = get_mask(mask)
        m_tu = get_mask(mask, 'tumor')
        m_or = get_mask(mask, 'organ')

        m_bg = np.expand_dims(m_bg, axis = -1)
        m_tu = np.expand_dims(m_tu, axis = -1)
        m_or = np.expand_dims(m_or, axis = -1)

        m_fn = np.concatenate([m_bg, m_or, m_tu], axis = -1)

        # mask = np.array(cv2.imread(mask_path)[:,:,0]) # shape = (512, 512)

        if self.transform is not None:
            augmentations = self.transform(image = image, mask = m_fn)
            image = augmentations['image']
            mask = augmentations['mask']

        return image, mask

