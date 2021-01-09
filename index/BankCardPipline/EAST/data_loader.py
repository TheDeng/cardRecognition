import torch
import os
import cv2
from .helper import *
import torch.utils.data as data
import numpy as np

class MyDataset(data.Dataset):
    def __init__(self, data_path, input_size):
        # TODO
        # 1. Initialize file paths or a list of file names. 
        self.imgs = []
        self.labels = []
        img_postfixs = ['jpg', 'png', 'jpeg', 'JPG']
        for f in os.listdir(data_path):
            tmp = f.split('.')
            if len(tmp) > 1 and tmp[1] in img_postfixs:
                label_path = os.path.join(data_path, '%s.txt'%tmp[0])
                img_path = os.path.join(data_path, f)
                if os.path.exists(label_path) and os.path.exists(img_path):
                    self.imgs.append(img_path)
                    self.labels.append(label_path)
        self.input_size = input_size
                
    def __getitem__(self, index):
        # TODO
        # 1. Read one data from file (e.g. using numpy.fromfile, PIL.Image.open).
        # 2. Preprocess the data (e.g. torchvision.Transform).
        # 3. Return a data pair (e.g. image and label).
        im = cv2.imread(self.imgs[index])
        polys = load_annoataion(self.labels[index])
        new_h, new_w, _ = im.shape
        polys = check_and_validate_polys(polys, new_h, new_w)
        max_h_w_i = np.max([new_h, new_w, self.input_size])
        im_padded = np.zeros((max_h_w_i, max_h_w_i, 3), dtype=np.uint8)
        im_padded[:new_h, :new_w, :] = im.copy()
        im = cv2.resize(im_padded, dsize=(self.input_size, self.input_size))
        resize_ratio_x = self.input_size/float(new_w)
        resize_ratio_y = self.input_size/float(new_h)
        polys[:, :, 0] *= resize_ratio_x
        polys[:, :, 1] *= resize_ratio_y
        score_map, geo_map = generate_rbox(im.shape[:2], polys)
        im = im[:, :, ::-1].astype(np.float32).transpose(2,0,1)
        score_map = score_map[np.newaxis,::4,::4].astype(np.float32)
        geo_map = geo_map[::4,::4,:].astype(np.float32).transpose((2,0,1))
        return im, score_map, geo_map


    def __len__(self):
        # You should change 0 to the total size of your dataset.
        return len(self.imgs)
