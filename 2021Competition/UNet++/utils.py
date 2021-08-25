import argparse
import torch
import numpy as np

def str2bool(v):
    if v.lower() in ['true', 1]:
        return True
    elif v.lower() in ['false', 0]:
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def save_checkpoint(state, filename = 'unet_checkpoint.pth.tar'):
    root = '/content/drive/Shareddrives/Kaggle_MOAI2021/checkpoints'
    print("=>saving checkpoint...")
    torch.save(state, root + '/' + filename)

def load_checkpoint(checkpoint, model):
    print("=>loading checkpoint...")
    model.load_state_dict(checkpoint['state_dict'])

def get_mask(mask, type = None):
    mask_1 = mask < 0.00784314
    mask_2 = mask >= 0.00392157
    mask_3 = np.logical_and(mask_1, mask_2)
    if type == 'tumor':
        mask = (mask >= 0.00784314).astype(float)
    elif type == 'organ':
        mask = mask_3.astype(float)
    elif type == "all":
        mask = (mask!=0).astype(float)
    else:
        mask = (mask < 0.00392157).astype(float)
    return mask

def get_loaders(train_dir, train_maskdir, batch_size, train_transform, valid_transform, num_workers = 1, pin_memory = True):
    # 매번 validataion dataset을 새롭게 설정해 준다고 하면 (전체 dataset에서 비율을 정해서 해줌)
    train_ds = MOAIDataset(
        image_dir = train_dir,
        mask_dir = train_maskdir,
        transform = train_transform
    )
    
    train_loader = DataLoader(
        train_ds,
        batch_size = batch_size, shuffle = True,
        num_workers = num_workers, pin_memory=pin_memory
    )

    valid_ds = MOAIDataset(
        image_dir = train_dir,
        mask_dir = train_maskdir,
        transform = valid_transform
    )

    valid_loader = DataLoader(
        valid_ds, 
        batch_size = batch_size, shuffle = True,
        num_workers = num_workers, pin_memory = pin_memory
    )
    print(len(train_ds), len(valid_ds))
    return train_loader, valid_loader

def check_accuracy(loader, scheduler, model, device = "cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()  # unet모델의 가중치를 학습하는게 아니라 evaluation 단계에 들어갔음을 의미하는 명령어

    with torch.no_grad():
        for image, mask in loader:
            image = image.to(device)
            mask = mask.permute(0, 3, 1, 2).to(device)
            print('making predictions...')
            predictions = torch.sigmoid(model(image.float()))
            print("prediction made...")

            result = (predictions > 0.5)[0].float().cpu().permute(1, 2, 0).detach().numpy()
            visualize_mask(to_cpu(predictions)[0].permute(1, 2, 0).numpy())
            visualize_mask(result)

            predictions = (predictions > 0.5).float()
            num_correct += (predictions == mask).sum()
            num_pixels += torch.numel(predictions)
            dice_score += (2 * (predictions * mask).sum()) / ((predictions + mask).sum() + 1e-9)
            
    scheduler.step(dice_score/len(loader))
        
    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.3f}")
    print(f"Got DICE Score : {dice_score/len(loader)}")
    model.train()