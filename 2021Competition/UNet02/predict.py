import os, glob
import numpy as np
import torch
import PIL.Image as Image
import torch.nn.functional as F
from torchvision import transforms

from dataloader import MOAIDataset
from unet import UNET

def mask_to_image(mask):
    if mask.ndim == 2:
        return Image.fromarray((mask * 255).astype(np.uint8))
    else:
        return Image.fromarray((np.argmax(mask, axis = 0)*255).astype(np.uint8))

def predict_img(model, img, device, n_classes):
    # test data중에서 이미지를 하나씩 넣어준다.
    model.eval()
    with torch.no_grad():
        img = torch.from_numpy(np.expand_dims(img, axis = 0)).to(device)

        output = model(img)

        if n_classes == 1:
            probs = torch.sigmoid(output)[0]
        else:
            probs = F.softmax(output, dim = 1)[0]
    
        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((512, 512)),
            transforms.ToTensor()
        ])

        final_mask = tf(probs.cpu()).squeeze()
    
    if n_classes == 1:
        return (final_mask > 0.5).numpy()
    else:
        return F.one_hot(final_mask.argmax(dim = 0), n_classes).permute(2, 0, 1).numpy()

def submit_encode(mask):
    m_bg, m_org, m_tu = mask[:,:,0], mask[:,:,1], mask[:,:,2]
    m_not_org = (m_org == 0)
    m_tu2 = np.logical_and(m_bg, m_not_org)
    x = np.zeros((mask.shape[0], mask.shape[1], 1))

    x[:,:,0][m_org == 1] = 1
    x[:,:,0][m_tu == 1] = 2
    x[:,:,0][m_tu2 == 1] = 2

    return x

if __name__ == "__main__":
    model = UNET(ch_in = 3, ch_out = 3).to(DEVICE)
    # load checkpoints to model
    load_checkpoint(torch.load('/content/drive/Shareddrives/Kaggle_MOAI2021/checkpoints/UNET_checkpoint.pth.tar'), model)

    test_dir = '/content/drive/Shareddrives/Kaggle_MOAI2021/data/test/DICOM'
    test_ids = os.listdir(test_dir)
    test_images = np.array(sorted(glob(test_dir + '/*/*')))

    preds_test = []
    test_transform = A.Compose([
        A.Normalize(
            mean = [0.0,0.0,0.0],
            std = [1.0,1.0,1.0],
            max_pixel_value = 255.0
        ),
        ToTensorV2()
    ])

    for n, path in tqdm(enumerate(test_images), total = len(test_images)):
        img = read_dicom(path, 100, 50)
        img = test_transform(image = img)['image']

        result = predict_img(model, img, DEVICE, 3)
        submit = submit_encode(result)
        preds_test.append(submit)
        plt.figure(figsize = (9,9))
        plt.subplot(1, 3, 1)
        plt.imshow(img)
        plt.subplot(1, 3, 2)
        plt.imshow(mask_to_image(result))
        plt.subplot(1, 3, 3)
        plt.imshow(submit)
        plt.show()
        print(np.unique(submit), (submit == 2).sum())
        