import torch
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from torchvision import transforms
from model import CRNN
from matplotlib import pyplot as plt



def img_preprocess(img):
    img = img_reshape(img, (64, 256))
    pil_img = Image.fromarray(img)
    ten_img = transforms.PILToTensor()(pil_img)
    return ten_img


def img_reshape(img, tar_size)->np.array:
    (h,w) = img.shape
    fin_img = np.ones(tar_size)*255
    if w>256:
        img = img[:,:tar_size[1]]
    if h>64:
        img = img[:tar_size[0],:]
    fin_img[:h,:w] = img
    return fin_img

if __name__ == '__main__':
    model = CRNN()
    model.load_state_dict(torch.load('OCR_CTC/model.pth'))
    model.eval()
    path = str(input('File path'))
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    plt.imshow(img, cmap='gray')
    img = img_preprocess(img=img)
    img = torch.unsqueeze(img, 0)
    preds, _ = model(img)
    print(preds)

