import torch
import numpy as np
import cv2
from PIL import Image

from torch.uitls import data
from torchvision import tranforms

class WritingDataGen(data.Dataset):
    def __init__(self, img_dataframe, tar_size, img_dir, token_dict):
        self.img_dataframe = img_dataframe
        self.tar_size = tar_size
        self.img_dir = img_dir
        self.transformer = transforms.Compose([
            transforms.PILToTensor(),
        ])
        self.token_dict = token_dict
    
    def __len__(self):
        return int(len(self.img_dataframe)/5)

    def img_preprocess(self, img)->np.array:
        (h,w) = img.shape
        fin_img = np.ones(self.tar_size)*255
        if w>256:
            img = img[:,:self.tar_size[1]]
        if h>64:
            img = img[:self.tar_size[0],:]
        fin_img[:h,:w] = img
        return fin_img
    
    def tokenizer(self, inputs)-> np.array:
        token_list = []
        for i in inputs:
            try:
                token_list.append(self.token_dict[i])
            except:
                continue
        
        token_arr = np.zeros(64)
        token_arr[:len(token_list)] = token_list
        return token_arr
    
    def __getitem__(self, idx):
        try:
            img_file = self.img_dataframe.FILENAME[idx]
            img_iden = self.img_dataframe.IDENTITY[idx]
        except:
            img_file = self.img_dataframe.FILENAME[idx+1]
            img_iden = self.img_dataframe.IDENTITY[idx+1]
        img_path = os.path.join(self.img_dir, img_file)
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = self.img_preprocess(image)
        image = Image.fromarray(image)
        image = self.transformer(image)
        target = self.tokenizer(img_iden)

        input_len = torch.IntTensor([image.shape[2]//8])
        tar_len = torch.IntTensor([len(img_iden)])
        
        return image, [target, input_len, tar_len]
    