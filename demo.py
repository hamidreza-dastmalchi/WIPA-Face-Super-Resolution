from Dataset_LH import Dataset_LH
from torch.utils.data import DataLoader
from modules import GeneratorNet2
import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import argparse


parser=argparse.ArgumentParser(description="input parameters for WIDA algorithm")
parser.add_argument("--disable_cuda", type=bool, default=False, help="set True if you want to disable cuda")
parser.add_argument("--wavelet_integrated", type=bool, default=True, help="set True if you want to integrate wavelet coefficients")
parser.add_argument("--scale", type=int, default=8, help="the upscaling factor")
parser.add_argument("--gt_root",  default="./sample_images/gt", help="the path of ground truth sample images")
parser.add_argument("--lr_root",  default="./sample_images/lr", help="the path of low-resolution sample images")
parser.add_argument("--sr_root",  default="./sample_images/sr", help="the path of super-resolved sample images")
parser.add_argument("--pretrained_folder", default="./pretrained")
parser.add_argument("--wi_net",  default="gen_net_8x", help="the file name of the pre-trained wavelet-integrated network")
args=parser.parse_args()

if torch.cuda.is_available() and not args.disable_cuda:
    device=torch.device("cuda")
else:
    device=torch.device("cpu")

sr_net=GeneratorNet2(args.scale).to(device)
Upsample = nn.Upsample(128, mode="bilinear", align_corners=True)
sr_net_file=os.path.join(args.pretrained_folder, args.wi_net)
sr_net.load_state_dict(torch.load(sr_net_file))
sr_net.eval()

dataset=Dataset_LH(args.gt_root, None, args.scale)
dataloader=DataLoader(dataset, batch_size=1, shuffle=False)


def main():
    for counter, data in enumerate(dataloader):
        high_image, low_image, name = data
        l_img=2*(low_image-0.5)
        l_img=l_img.cuda()
        with torch.no_grad():
            sr_img, _=sr_net(l_img)
            sr_img=(sr_img/2+0.5).cpu()
        name1 = "LR"+name[0][2:]
        name1=os.path.join(args.lr_root,name1)
        low_image=Upsample(low_image)
        low_image=(low_image.squeeze(0).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        low_image=cv2.cvtColor(low_image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(name1, low_image)
        sr_img = (sr_img.squeeze(0).numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        sr_img = cv2.cvtColor(sr_img, cv2.COLOR_BGR2RGB)
        name2 = "SR"+name[0][2:]
        name2=os.path.join(args.sr_root,name2)
        cv2.imwrite(name2, sr_img)

if __name__=="__main__":
    main()
