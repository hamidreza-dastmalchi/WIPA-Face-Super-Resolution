from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import skimage.transform as trf
from PIL import Image
import numpy as np
import cv2
import torch
import os



class Dataset_LH(Dataset):
    def __init__(self, image_folder, transform, scale):
        super(Dataset_LH).__init__()
        self.ToTensor = ToTensor()
        self.image_folder = image_folder
        self.image_list = os.listdir(image_folder)
        self.transform = transform
        self.scale = scale

    def __getitem__(self, item):
        image_name = self.image_list[item]
        image_full_name = os.path.join(self.image_folder, image_name)
        image = Image.open(image_full_name)
        if self.transform!=None:
            high_image = self.transform(image)
        else:
            high_image=image
        high_image_tolow = np.asarray(high_image, dtype=np.float32)
        if high_image_tolow.max() >= 2:
            high_image_tolow = high_image_tolow / 255
        if self.scale != None:
            low_image = trf.rescale(high_image_tolow, 1/(self.scale), multichannel=True, anti_aliasing=True, mode="reflect")
            low_image = self.ToTensor(low_image)
        else:
            low_image = 0
        high_image=self.ToTensor(high_image)
        return high_image, low_image, image_name

    def __len__(self):
        return(len(self.image_list))




class LFWDataset(torch.utils.data.Dataset):
    def __init__(self, pair_path, data_path):
        super(LFWDataset, self).__init__()
        self.data_path = data_path
        with open(pair_path) as f:
            pairs_lines = f.readlines()[1:]
        self.pairs_lines = pairs_lines

    def __getitem__(self, index):
        p = self.pairs_lines[index].replace('\n', '').split('\t')
        if 3 == len(p):
            sameflag = np.int32(1).reshape(1)
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[2]))
        if 4 == len(p):
            sameflag = np.int32(0).reshape(1)
            name1 = p[0] + '/' + p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
            name2 = p[2] + '/' + p[2] + '_' + '{:04}.jpg'.format(int(p[3]))
        img1 = cv2.imread(os.path.join(self.data_path, name1))
        img2 = cv2.imread(os.path.join(self.data_path, name2))
        ## Resize second image
        #img2 = cv2.resize(img2, None, fx=1 / self.scale, fy=1 / self.scale,
        #                 interpolation=cv2.INTER_CUBIC)
        img1 = ToTensor()(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
        img2 = ToTensor()(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
        return img1, img2, torch.LongTensor(sameflag)

    def __len__(self):
        return len(self.pairs_lines)


