from modules import GeneratorNet, GeneratorNet2, DiscriminatoreNet
from modules import VggFeatures, weights_init
from modules import wavelet_packet
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from Dataset_LH import Dataset_LH
import torchvision.transforms as transforms
import torchvision.models as models
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
#commit1

parser=argparse.ArgumentParser(description="input parameters for WIDA algorithm")
parser.add_argument("--epoch_num", type=int, default=200, help="number of training epochs")
parser.add_argument("--batch_size", type=int, default=32, help="batch size of training procedure")
parser.add_argument("--num_workers", type=int, default=2, help="number of cpu workers for data loading")
parser.add_argument("--disable_cuda", type=bool, default=False, help="set True if you want to disable cuda")
parser.add_argument("--lr", type=float, default=0.0001, help="learning rate for training procedure")
parser.add_argument("--wavelet_integrated", type=bool, default=True, help="set True if you want to integrate wavelet coefficients")
parser.add_argument("--GAN", type=bool, default=True, help="set True if you want to include GAN adversarial training")
parser.add_argument("--scale", type=int, default=8, help="the upscaling factor")
parser.add_argument("--adv_weight", type=float, default=0.001, help="the weight of the adversarial loss function")
parser.add_argument("--id_weight", type=float, default=0.005, help="the weight of the identity loss function")
parser.add_argument("--mse_weight", type=float, default=1.0, help="the weight of the mse loss function")
parser.add_argument("--vgg_weight", type=float, default=0.001, help="the weight of the vgg perceptual loss function")
parser.add_argument("--wavelet_weight", type=float, default=1.0, help="the weight of the wavelet loss function")
parser.add_argument("--beta1", type=float, default=0.5, help="the beta1 coefficient for Adam optimizer")
parser.add_argument("--beta2", type=float, default=0.999, help="the beta2 coefficient for Adam optimizer")
parser.add_argument("--decay_rate", type=float, default=0.5, help="the decay rate of learning rate")
parser.add_argument("--epochs_todecay", type=int, default=40, help="number of epochs to decay the learning rate")
parser.add_argument("--num_iter_tolog", type=int, default=50, help="number of iterations to log the performance")
parser.add_argument("--seed", type=int, default=123, help="the seed of random generator")
parser.add_argument("--train_root",  default="./data/train")
parser.add_argument("--test_root",  default="./data/test/celeba")
parser.add_argument("--checkpoints_root",  default="./checkpoints")
parser.add_argument("--pretrained_folder", default="./pretrained")
parser.add_argument("--base_net",  default="", help="the file name of the pre-trained baseline network")
parser.add_argument("--wi_net",  default="gen_net_8x", help="the file name of the pre-trained wavelet-integrated network")
parser.add_argument("--disc_net",  default="", help="the file name of the pre-trained discriminator network")
parser.add_argument("--sphere_net", default="sface.pth", help="the file name of the pre-trained sphere network")
parser.add_argument("--log_dir",  default="./logs")
args=parser.parse_args()
#########################################################################
if not (os.path.isdir(args.checkpoints_root)):
    os.mkdir(args.checkpoints_root)
writer=SummaryWriter(args.log_dir)
#writer logs the scalars and images in tensorboard
#########################################################################
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
c1=torch.linspace(0, 0.1, args.epoch_num)
c2=torch.linspace(0.1, 0, args.epoch_num)
c3=torch.linspace(0.2, 0.1, args.epoch_num)
#c1, c2 and c3 are coefficients to manipulate the discriminator label to
#avoid it from becomming over-confident
#########################################################################
#train data loader
train_dataset=Dataset_LH(args.train_root, None, args.scale)
train_dataloader=DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True, num_workers=args.num_workers)
num_batch=len(train_dataset)//(args.batch_size)
upsample=nn.Upsample(128, mode="bilinear", align_corners=True)
#########################################################################
#test dataloader
test_dataset=Dataset_LH(args.test_root, None, args.scale)
test_dataloader=DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)
test_dataloader_iter=iter(test_dataloader)
high_image_eval, low_image_eval, _ = next(test_dataloader_iter)
high_image_eval=high_image_eval[0:16, :, : , :]
low_image_eval=low_image_eval[0:16, :, : , :]
high_image_eval_grid=make_grid(high_image_eval, nrow=4, padding=4)
low_image_eval_up=upsample(low_image_eval)
low_image_eval_up_grid=make_grid(low_image_eval_up, nrow=4, padding=4)
low_image_eval=2*(low_image_eval-.5)
low_image_eval=low_image_eval.cuda()
writer.add_image("Original Eval Images", high_image_eval_grid, 0)
writer.add_image("Input Eval Images", low_image_eval_up_grid, 0)
#########################################################################
#mean and std to normalize images before feeding to vgg feature extractor
mean1=torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).unsqueeze(1).unsqueeze(2).unsqueeze(0).cuda()
std1=torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).unsqueeze(1).unsqueeze(2).unsqueeze(0).cuda()
normalize = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])
vgg_feature_extractor=VggFeatures(models.vgg19(pretrained=True)).cuda()
#########################################################################
if torch.cuda.is_available() and not args.disable_cuda:
    device=torch.device("cuda")
else:
    device=torch.device("cpu")
#if wavelet_integrated is True the wavelet-integrated network is chosen
#otherwise, the baseline network is selected
if args.wavelet_integrated:
    gen_net = GeneratorNet2(scale=args.scale).to(device)
    model_file=os.path.join(args.pretrained_folder, args.wi_net)
    if os.path.isfile(model_file):
        gen_net.load_state_dict(torch.load(model_file))
    else:
        weights_init(gen_net)
else:
    gen_net=GeneratorNet(scale=args.scale).to(device)
    model_file=os.path.join(args.pretrained_folder, args.baseline_net)
    if os.path.isfile(model_file):
        gen_net.load_state_dict(torch.load(model_file))
    else:
        weights_init(gen_net)

#discriminator to be used in GAN framework
disc_net=DiscriminatoreNet().to(device)
model_file=os.path.join(args.pretrained_folder, args.disc_net)
if os.path.isfile(model_file):
    disc_net.load_state_dict(model_file)
else:
    weights_init(disc_net)

#sface of sphereface to be used as identity vector extractor
fnet = getattr(FNet, 'sface')().to(device)
fnet.load_state_dict(torch.load(os.path.join(args.pretrained_folder, args.sphere_net)))
fnet.eval()
#########################################################################
disc_optim=optim.Adam(disc_net.parameters(), lr=args.lr, betas=[args.beta1, args.beta2])
gen_optim=optim.Adam(gen_net.parameters(), lr=args.lr, betas=[args.beta1, args.beta2])
disc_schedul=lr_scheduler.StepLR(disc_optim, args.epochs_todecay, args.decay_rate)
gen_schedul=lr_scheduler.StepLR(gen_optim, args.epochs_todecay, args.decay_rate)
BCE_Loss=nn.BCELoss(reduction="mean").to(device)
MSE_Loss=nn.MSELoss(reduction="mean").to(device)
MAE_Loss=nn.L1Loss(reduction="mean").to(device)
cosine_loss=nn.CosineSimilarity().to(device)
#########################################################################
def main():
    str_time=time.time()
    counter=0
    for ep in range(args.epoch_num):
        if ep<1:
            alpha1 = 0
            alpha2 = 0
            alpha3 = 1
            alpha4 = 0
            alpha5 = 0
        else:
            alpha1 = args.adv_weight
            alpha2 = args.id_weight
            alpha3 = args.mse_weight
            alpha4 = args.vgg_weight
            alpha5 = args.wavelet_weight
        if ep%10==9:
            gen_net.eval()
            torch.save(gen_net.state_dict(), os.path.join(args.checkpoints_root,"gen_net_"+str(args.scale)+"_x_{}".format(ep+1)))
            gen_net.train()
            disc_net.eval()
            torch.save(disc_net.state_dict(), os.path.join(args.checkpoints_root,"disc_net_"+str(args.scale)+"_x_{}".format(ep+1)))
            disc_net.train()
        gen_adv_loss = 0
        gen_id_loss = 0
        gen_mse_loss = 0
        gen_vgg_loss = 0
        gen_wavelet_loss = 0
        gen_total_loss = 0
        disc_real_loss = 0
        disc_fake_loss = 0
        disc_total_loss = 0
        gen_net.train()
        disc_net.train()
        for count, data in enumerate(train_dataloader):
            high_image , low_image, _ = data
            #low_image = low_generation(high_image, args.scale)
            if args.wavelet_integrated:
                wp=wavelet_packet(high_image, args.scale)
            low_image = 2*(low_image-.5)
            low_image = low_image.detach().to(device)
            high_image = 2*(high_image-.5)
            high_image = high_image.to(device)
            #####################################################################################################
            sr_image = gen_net(low_image)
            if args.wavelet_integrated:
                sr_image, sr_wavelets=sr_image
            if ep>=2:
                #after training for two epochs, the discriminator starts to be trained
                d_fake = disc_net(sr_image.detach())
                d_real = disc_net(high_image)
                dloss_fake = BCE_Loss(d_fake, c3[ep]* torch.rand_like(d_fake, dtype=d_fake.dtype).cuda() )
                dloss_real = BCE_Loss(d_real,  0.8 + c1[ep] + (0.2-c1[ep])* torch.rand_like(d_real, dtype=d_real.dtype).cuda())
                dloss = dloss_real + dloss_fake
                disc_total_loss += dloss.item()
                disc_real_loss += dloss_real.item()
                disc_fake_loss += dloss_fake.item()
                disc_optim.zero_grad()
                dloss.backward()
                disc_optim.step()
            #####################################################################################################
            wavelet_loss=0
            if args.wavelet_integrated:
                for indw in range(len(sr_wavelets)):
                    wavelet_loss+=MAE_Loss(sr_wavelets[indw], wp[indw].to(device))/(2**(2*indw))
                wavelet_loss=wavelet_loss/len(sr_wavelets)
            #after 4 epochs, the adversarial loss is considered in the generator
            if args.GAN and ep>=4:
                d_fake_g = disc_net(sr_image)
                gadv_loss = BCE_Loss(d_fake_g, torch.ones_like(d_fake).to(device))
            else:
                gadv_loss = torch.zeros((1), dtype=torch.float32).to(device)
            gmse_loss = MSE_Loss(sr_image / 2 + 0.5, high_image / 2 + 0.5)
            fake_feature = vgg_feature_extractor(((sr_image / 2 + 0.5) - mean1) / std1)
            real_feature = vgg_feature_extractor(((high_image / 2 + 0.5) - mean1) / std1)
            gvgg_loss = MSE_Loss(real_feature, fake_feature)
            sr_image_crop = sr_image[:, :, 9:120, 17:112]
            high_image_crop = high_image[:, :, 9:120, 17:112]
            sr_identity = fnet(sr_image_crop * 127.5 / 128)
            hr_identity = fnet(high_image_crop * 127.5 / 128)
            gid_loss = 1 - cosine_loss(sr_identity, hr_identity)
            gid_loss = gid_loss.mean()
            gen_optim.zero_grad()
            disc_optim.zero_grad()
            #gadv_loss is the generator adversarial loss
            #gid_loss is the identity loss
            #gmse_loss is the pixel-wise mse loss
            #gvgg_loss is the vgg perceptual loss
            #wavelet_loss is the wavelet loss
            if args.wavelet_integrated:
                gen_loss = alpha1 * gadv_loss + alpha2 * gid_loss + alpha3 * gmse_loss + alpha4 * gvgg_loss + alpha5 * wavelet_loss
            else:
                gen_loss = alpha1 * gadv_loss + alpha2 * gid_loss + alpha3 * gmse_loss + alpha4 * gvgg_loss
            gen_total_loss += gen_loss.item()
            gen_adv_loss += gadv_loss.item()
            gen_id_loss += gid_loss.item()
            gen_mse_loss += gmse_loss.item()
            gen_vgg_loss += gvgg_loss.item()
            if args.wavelet_integrated:
                gen_wavelet_loss += wavelet_loss.item()
            gen_loss.backward()
            gen_optim.step()
            #####################################################################################################
            if count % args.num_iter_tolog == args.num_iter_tolog - 1:
                end_time = time.time()
                gen_net.eval()
                dur_time=end_time-str_time
                with torch.no_grad():
                    if args.wavelet_integrated:
                        sr_image_eval, sr_wavelets_eval = gen_net(low_image_eval)
                        for indw in range(len(sr_wavelets_eval)):
                            #logging the predicted wavelet coefficients and also SR image in tensorboard
                            sr_wavelets_eval_h = torch.abs(sr_wavelets_eval[indw][:,0,:,:].cpu().unsqueeze(1))
                            sr_wavelets_eval_v = torch.abs(sr_wavelets_eval[indw][:, 1, :, :].cpu().unsqueeze(1))
                            sr_wavelets_eval_d = torch.abs(sr_wavelets_eval[indw][:, 2, :, :].cpu().unsqueeze(1))
                            sr_wavelets_eval_h =sr_wavelets_eval_h/torch.max(sr_wavelets_eval_h)
                            sr_wavelets_eval_v =sr_wavelets_eval_v/torch.max(sr_wavelets_eval_v)
                            sr_wavelets_eval_d =sr_wavelets_eval_d/torch.max(sr_wavelets_eval_d)
                            sr_wavelets_eval_h_grid = make_grid(sr_wavelets_eval_h, nrow=4, padding=4)
                            sr_wavelets_eval_v_grid = make_grid(sr_wavelets_eval_v, nrow=4, padding=4)
                            sr_wavelets_eval_d_grid = make_grid(sr_wavelets_eval_d, nrow=4, padding=4)
                            writer.add_image("super resolved horizontal wavelet in scale "+str(indw), sr_wavelets_eval_h_grid, counter)
                            writer.add_image("super resolved vertical wavelet in scale " + str(indw), sr_wavelets_eval_v_grid,
                                     counter)
                            writer.add_image("super resolved diagonal wavelet in scale " + str(indw), sr_wavelets_eval_d_grid,
                                     counter)
                    else:
                        sr_image_eval = gen_net(low_image_eval)
                    sr_image_eval = sr_image_eval.cpu()
                    eval_sr = sr_image_eval / 2 + .5
                eval_sr_grid = make_grid(eval_sr, nrow=4, padding=4)
                #logging the training loss terms
                writer.add_image("super resolved images", eval_sr_grid, counter)
                writer.add_scalar("generator adversarial loss", gen_adv_loss/args.num_iter_tolog, counter)
                writer.add_scalar("generator identity loss", gen_id_loss/args.num_iter_tolog, counter)
                writer.add_scalar("generator mse loss", gen_mse_loss/args.num_iter_tolog, counter)
                writer.add_scalar("generator vgg loss", gen_vgg_loss/args.num_iter_tolog, counter)
                writer.add_scalar("generator wavelet loss", gen_wavelet_loss/args.num_iter_tolog, counter)
                writer.add_scalar("discriminator real loss", disc_real_loss/args.num_iter_tolog, counter)
                writer.add_scalar("discriminator fake loss", disc_fake_loss/args.num_iter_tolog, counter)
                writer.add_scalar("discriminator total loss", disc_total_loss/args.num_iter_tolog, counter)
                print("epoch {0:03d}/{1:03d} \t iter {2:04d}/{3:04d} \t gen adv loss {4:0.4f} \t gen id loss {5:0.4f} "
                  "\t gen mse loss {6:0.4f} \t gen vgg loss {7:0.4f} \t dis adv loss {8:0.04f} \t time: {9:0.02f}".
                    format(ep+1, args.epoch_num, count+1, num_batch, gen_adv_loss/args.num_iter_tolog, gen_id_loss/args.num_iter_tolog,
                        gen_mse_loss/args.num_iter_tolog, gen_vgg_loss/args.num_iter_tolog, disc_total_loss/args.num_iter_tolog, dur_time))
                gen_adv_loss = 0
                gen_id_loss = 0
                gen_mse_loss = 0
                gen_vgg_loss = 0
                gen_wavelet_loss = 0
                gen_total_loss = 0
                disc_real_loss = 0
                disc_fake_loss = 0
                disc_total_loss = 0
                counter += 1
                gen_net.train()
                str_time = time.time()
        disc_schedul.step()
        gen_schedul.step()

if __name__=="__main__":
    main()






















