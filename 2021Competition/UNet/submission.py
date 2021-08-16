# submission을 위한 prediction값을 저장해 준다.
# 출력값에 대해서 sigmoid 를 이용해서 0과 1사이의 값으로 바꾸어 주고
# my_encode()함수를 사용해서 
# channel별로 0, 1, 2의 값으로 저장해 준다.
import torch, os
import numpy as np
from glob import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import os

model = UNET(ch_in = 3, ch_out = 3).to(DEVICE)
load_checkpoint(torch.load('/content/drive/Shareddrives/Kaggle_MOAI2021/checkpoints/unet_checkpoint.pth.tar'), model)



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

for n, path in tqdm(enumerate(test_images), total=len(test_images)):
    id = path.split('/')[-2]
    inp = read_dicom(path, 100, 50)
    inp = test_transform(image = inp)['image']
    inp = torch.from_numpy(np.expand_dims(inp, axis = 0)).to(DEVICE)
    out = torch.sigmoid(model(inp))
    out = out.detach().cpu().permute(0, 2, 3, 1)[0]
    #plt.imshow(out)
    #plt.show()
    #out_1, out_2, out_3 = out[:,:,0], out[:,:,1], out[:,:,2]
    #plt.imshow(out_1, cmap = 'gray')
    #plt.show()
    #plt.imshow(out_2, cmap = 'gray')
    #plt.show()
    #plt.imshow(out_3, cmap = 'gray')
    #plt.show()
    out = my_encode(out)
    out = pred_encode(out)
    #plt.imshow(out[:,:,0])
    #plt.show()
    preds_test.append(out)
    
    # 배경 -> 0
    # out_1에서 1로 표시되는 부분이 배경
    # out_1에서 0으로 표시되는 부분이 신장
    # out_2에서 1로 표시되는 부분이 신장 + 종양
    # out_3에서 1로 표시되는 부분이 종양
    
 
import pandas as pd
# rlencoding방법을 적용해서 csv파일로 만들어 제출이 가능하도록 한다..
preds_string = []
for i in tqdm(range(0, len(preds_test), 64)):
    sample = preds_test[i:i+64].copy()
    for label_code in [1,2]:
        tmp=[]
        for s in sample:
            s = np.equal(s, label_code).flatten()*1
            tmp+=s.tolist()
        enc = rle_to_string(rle_encode(np.array(tmp)))

        preds_string.append(enc)

sample_submission = pd.read_csv('/content/drive/Shareddrives/Kaggle_MOAI2021/data/sample_submission.csv')
sample_submission['EncodedPixels'] = preds_string
sample_submission.to_csv('/content/drive/Shareddrives/Kaggle_MOAI2021/checkpoints/submission.csv', index=False)