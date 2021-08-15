import torch
import torchvision
import numpy as np
from dataset import MOAIDataset
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def get_mask(mask, type = None):
    mask_1 = mask < 0.007843414
    mask_2 = mask >= 0.00392157
    mask_3 = np.logical_and(mask_1, mask_2)
    if type == 'tumor':
        mask = mask_1.astype(float)
    elif type == 'organ':
        mask = mask_3.astype(float)
    else:
        mask = mask_2.astype(float)
    return mask

def my_encode(mask):
    x = np.zeros((mask.shape[0], mask.shape[1], 1))
    x[mask[:,:,0] > 0.5] = 0 # bg
    x[mask[:,:,1] > 0.5] = 1 # organ
    x[mask[:,:,2] > 0.5] = 2 # tumor

    return x

def save_checkpoint(state, filename = 'unet_checkpoint.pth.tar'):
    print("=>saving checkpoint...")
    torch.save(state, filename)

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
    

def check_accuracy(loader, model, device = "cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()  # unet모델의 가중치를 학습하는게 아니라 evaluation 단계에 들어갔음을 의미하는 명령어

    with torch.no_grad():
        for image, mask in loader:
            image = image.to(device)
            mask = mask.permute(0, 3, 1, 2).to(device)
            print(image.shape)
            print(mask.shape)
            print('making predictions...')
            predictions = torch.sigmoid(model(image.float()))
            print("prediction made...")

            plt.imshow(predictions[0].permute(1, 2, 0).detach().numpy())
            plt.show()

            predictions = (predictions > 0.5).float()
            num_correct += (predictions == mask).sum()
            num_pixels += torch.numel(predictions)
            dice_score += (2 * (predictions * mask).sum()) / ((predictions + mask).sum() + 1e-9)
        
    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.3f}")
    print(f"Got DICE Score : {dice_score/len(loader)}")
    model.train()

def save_predictions_as_imgs(loader, model, folder = "predicted_masks/", device = 'cuda'):
    model.eval()

    for index, (image, mask) in enumerate(loader):
        image = image.to(device = device)
        with torch.no_grad():
            prediction = (torch.sigmoid(model(image)))
            prediction = (prediction < 0.33).float()
        torchvision.utils.save_image(prediction, f"{folder}/pred_{index}.png")
        torchvision.utils.save_image(mask.unsqueeze(1), f"{folder}{index}.png")

    model.train()

