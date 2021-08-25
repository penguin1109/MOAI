import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from dataloader import MOAIDataset

def get_mask(mask, type = None):
    mask_1 = mask < 0.00784314
    mask_2 = mask >= 0.00392157
    mask_3 = np.logical_and(mask_1, mask_2)
    if type == 'tumor':
        mask = (mask >= 0.00784314).astype(float)
    elif type == 'organ':
        mask = mask_3.astype(float)
    else:
        mask = (mask == 0).astype(float)
    return mask

def visualize_masks(mask):
  m_1, m_2, m_3 = mask[:,:,0], mask[:,:,1], mask[:,:,2]
  masks = [mask, m_1, m_2, m_3]
  
  plt.figure(figsize = (10,10))
  for i in range(4):
    plt.subplot(1,4, i+1)
    plt.imshow(masks[i])
  plt.show()
  
def to_cpu(tensor):
  return tensor.detach().cpu()

def save_checkpoint(state, filename = 'UNET_checkpoint.pth.tar'):
    root = '/content/drive/Shareddrives/Kaggle_MOAI2021/checkpoints'
    print("=>saving checkpoint...")
    torch.save(state, root + '/' + filename)

def load_checkpoint(checkpoint, model):
    print("=>loading checkpoint...")
    model.load_state_dict(checkpoint['state_dict'])

def get_loaders(train_dir, train_maskdir, batch_size, train_transform, valid_transform, rate, num_workers = 1, pin_memory = True):
    # 매번 validataion dataset을 새롭게 설정해 준다고 하면 (전체 dataset에서 비율을 정해서 해줌)
    train_ds = MOAIDataset(
        image_dir = train_dir,
        mask_dir = train_maskdir,
        transform = train_transform,
        rate = rate, valid = False
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size = batch_size, shuffle = True,
        num_workers = num_workers, pin_memory=pin_memory
    )

    valid_ds = MOAIDataset(
        image_dir = train_dir,
        mask_dir = train_maskdir,
        transform = valid_transform,
        rate = rate, valid = True
    )

    valid_loader = DataLoader(
        valid_ds, 
        batch_size = batch_size, shuffle = True,
        num_workers = num_workers, pin_memory = pin_memory
    )
    print(len(train_ds), len(valid_ds))
    return train_loader, valid_loader