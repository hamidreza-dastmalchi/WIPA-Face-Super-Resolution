from modules import GeneratorNet, GeneratorNet2, DiscriminatoreNet
from Dataset_LH import Dataset_LH
from Dataset_LH import LFWDataset
from evaluate import face_ver
from modules import VggFeatures, weights_init, low_generation
from modules import wavelet_packet
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from evaluate import wipa_psnr_ssim
from evaluate import fps_estimation
from evaluate import fid_estimation
import torchvision.transforms as transforms
import torchvision.models as models
import torchvision.datasets as datasets
from torchvision.utils import make_grid
from torch.optim import lr_scheduler
import torch.optim as optim
import torch.nn as nn
import numpy as np
import time
import torch
import random
import os
import argparse
import FNet






parser=argparse.ArgumentParser(description="input parameters for WIDA algorithm")
parser.add_argument("--batch_size", type=int, default=32, help="batch size of training procedure")
parser.add_argument("--disable_cuda", type=bool, default=False, help="set True if you want to disable cuda")
parser.add_argument("--wavelet_integrated", type=bool, default=True, help="set True if you want to integrate wavelet coefficients")
parser.add_argument("--scale", type=int, default=16, help="the upscaling factor")
parser.add_argument("--test_root",  default="./data/test/celeba")
parser.add_argument("--save_folder",  default="./results/celeba")
parser.add_argument("--save_flag", type=bool, default=False, help="set True if you want to save the super-resolved images")
parser.add_argument("--lfw_data", default="./data/lfw_pairs/aligned_lfw")
parser.add_argument("--lfw_pair_file", default="./data/lfw_pairs/lfw_pair_list/pairs.txt")
parser.add_argument("--pretrained_folder", default="./pretrained")
parser.add_argument("--base_net",  default="", help="the file name of the pre-trained baseline network")
parser.add_argument("--wi_net",  default="gen_net_16x", help="the file name of the pre-trained wavelet-integrated network")
parser.add_argument("--sphere_net", default="sface.pth", help="the file name of the pre-trained sphere network")
parser.add_argument("--metrics", type=str, nargs="+", default=["psnr", "ssim", "fid", "acc"])
args=parser.parse_args()





if torch.cuda.is_available() and not args.disable_cuda:
    device=torch.device("cuda")
else:
    device=torch.device("cpu")

if args.wavelet_integrated:
    gen_net = GeneratorNet2(scale=args.scale).to(device)
    model_file=os.path.join(args.pretrained_folder, args.wi_net)
    if os.path.isfile(model_file):
        gen_net.load_state_dict(torch.load(model_file))
    else:
        raise Exception("the pre-trained generator file is not found.")
else:
    gen_net=GeneratorNet(scale=args.scale).to(device)
    model_file=os.path.join(args.pretrained_folder, args.baseline_net)
    if os.path.isfile(model_file):
        gen_net.load_state_dict(torch.load(model_file))
    else:
        raise Exception("the pre-trained generator file is not found.")
gen_net.eval()
fnet = getattr(FNet, 'sface')().to(device)
fnet.load_state_dict(torch.load(os.path.join(args.pretrained_folder, args.sphere_net)))
fnet.eval()

if not os.path.isdir(args.test_root):
    raise Exception("the test folder does not exist.")
test_dataset=Dataset_LH(args.test_root, None, args.scale)
if len(test_dataset)==0:
    raise Exception("test folder is empty")
test_dataloader=DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)

lfw_pair_dataset=LFWDataset(args.lfw_pair_file, args.lfw_data)
lfw_data_loader = DataLoader(lfw_pair_dataset, batch_size=args.batch_size, shuffle=False, drop_last=False)




for i in range(len(args.metrics)):
    print(args.metrics[i])
    args.metrics[i] = args.metrics[i].lower()

if __name__ == '__main__':

    if "psnr" in args.metrics or "ssim" in args.metrics:
        psnr, ssim = wipa_psnr_ssim(test_dataloader, gen_net, args)
        print("psnr : {0:0.4f} \t ssim : {1:0.4f}".format(psnr, ssim))

    if "fps" in args.metrics:
        fps=fps_estimation(gen_net, args)
        print("fps : {0:0.4f}".format(fps))

    if "fid" in args.metrics:
        fid=fid_estimation(args)
        print("fid : {0:0.4f}".format(fid))

    if "acc" in args.metrics:
        tpr, fpr, acc = face_ver(lfw_data_loader, gen_net,fnet,args)
        print("verification rate : {0:0.4f}".format(acc))



