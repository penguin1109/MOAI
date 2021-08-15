import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from model import UNET
from utils import load_checkpoint, save_checkpoint, get_loaders, save_predictions_as_imgs, check_accuracy

# Hyperparameters
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LEARNING_RATE = 1e-4
BATCH_SIZE = 4
NUM_EPOCHS = 10
IMAGE_WIDTH = 512
IMAGE_HEIGHT = 512
LOAD_MODEL = False
PIN_MEMORY = True
NUM_WORKERS = 1
RATE = 0.7
TRAIN_IMAGE_DIR = '/content/drive/Shareddrives/Kaggle_MOAI2021/data/train/DICOM'
TRAIN_MASK_DIR = '/content/drive/Shareddrives/Kaggle_MOAI2021/data/train/Label'
TEST_IMAGE_DIR = '/content/drive/Shareddrives/Kaggle_MOAI2021/data/test'


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (image, mask) in enumerate(loop):
        images = image.to(device = DEVICE)
        masks = mask.permute(0, 3, 1, 2).float().to(device = DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(images)
            loss = loss_fn(predictions, masks)
        
        # backward
        optimizer.zero_grad() # optimizer의 gradient 값을 0으로 초기화
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss = loss.item())

def main():
    train_transform = A.Compose(
        [
            A.Rotate(limit = 35, p = 1.0),
            A.HorizontalFlip(p = 0.5),
            A.VerticalFlip(p = 0.1),
            A.Normalize(
                mean = [0.0, 0.0, 0.0],
                std = [1.0, 1.0, 1.0],
                max_pixel_value = 255.0
            ),
            ToTensorV2()
        ]
    )

    valid_transform = A.Compose(
        [
            A.Normalize(
                mean = [0.0, 0.0, 0.0],
                std = [1.0, 1.0, 1.0],
                max_pixel_value = 255.0
            ),
            ToTensorV2()
        ]
    )

    model = UNET(ch_in = 3, ch_out = 3).to(DEVICE)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    train_loader, valid_loader = get_loaders(
        TRAIN_IMAGE_DIR, TRAIN_MASK_DIR, BATCH_SIZE, 
        train_transform, valid_transform, rate = RATE,
        num_workers = NUM_WORKERS, pin_memory = PIN_MEMORY
    )


    if LOAD_MODEL:
        load_checkpoint(torch.load("unet_checkpoint.pth.tar"), model)

    #check_accuracy(valid_loader, model, device = DEVICE)
    scaler = torch.cuda.amp.GradScaler()

    for epoch in range(NUM_EPOCHS):
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # save model checkpoints
        checkpoint = {
            "state_dict" : model.state_dict(),
            "optimizer" : optimizer.state_dict()
        }
        save_checkpoint(checkpoint)

        # check accuracy
        check_accuracy(valid_loader, model, device = DEVICE)

        # save predicted images to new folder
        save_predictions_as_imgs(
            valid_loader, model, folder = 'predicted_masks/', device = DEVICE
        )

if __name__ == "__main__":
    main()