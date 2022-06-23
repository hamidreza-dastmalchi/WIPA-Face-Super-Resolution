from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from Dataset_LH import Dataset_LH
from torch.utils.data import DataLoader
from modules import KFold, eval_acc
from modules import find_best_threshold
from modules import low_generation
import torchvision.transforms as transforms
from fid import calculate_fid
import torch.nn as nn
import numpy as np
import torch
import os
import cv2
import time
from tqdm import tqdm



def wipa_psnr_ssim(test_dataloader, gen_net, args):
    if torch.cuda.is_available() and not args.disable_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    ssim_mean = 0
    psnr_mean = 0
    counter = 0
    save_folder = os.path.join(args.save_folder, str(args.scale)+"x")
    if not os.path.isdir(save_folder):
        os.makedirs(save_folder)
    print("psnr and ssim is calculated on test dataset, please wait...")
    for count, data in enumerate(tqdm(test_dataloader)):
        high_image, low_image, image_name = data
        low_image = low_image.to(device)
        low_image = 2 * (low_image - 0.5)
        with torch.no_grad():
            sr_image = gen_net(low_image)
            if args.wavelet_integrated:
                sr_image, sr_wavelet = sr_image
        sr_image = sr_image / 2 + 0.5
        sr_image = sr_image.cpu()
        for i in range(high_image.size()[0]):
            sr_image_i = sr_image[i]
            high_image_i = high_image[i]
            sr_image_i = sr_image_i.numpy().transpose(1, 2, 0)
            high_image_i = high_image_i.numpy().transpose(1, 2, 0)
            psnr = peak_signal_noise_ratio(high_image_i, sr_image_i)
            ssim = structural_similarity(high_image_i, sr_image_i, multichannel=True)
            if args.save_flag:
                sr_image_i=(sr_image_i*255).astype(np.uint8)
                sr_image_i=cv2.cvtColor(sr_image_i, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(save_folder, image_name[i]), sr_image_i)
            psnr_mean += psnr
            ssim_mean += ssim
            counter += 1
    ssim_mean = ssim_mean / counter
    psnr_mean = psnr_mean / counter
    return psnr_mean, ssim_mean


def fps_estimation(gen_net, args):
    if torch.cuda.is_available() and not args.disable_cuda:
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    test_dataset=Dataset_LH(args.test_root, None, args.scale)
    test_dataloader=DataLoader(test_dataset, batch_size=1, shuffle=False)
    str_time=time.time()
    for count, data in enumerate(test_dataloader):
        high_image, low_image, image_name = data
        low_image = low_image.to(device)
        low_image = 2 * (low_image - 0.5)
        with torch.no_grad():
            sr_image = gen_net(low_image)
            if args.wavelet_integrated:
                sr_image, sr_wavelet = sr_image
        sr_image = sr_image / 2 + 0.5
        if count==99:
            break
    end_time=time.time()
    time_dur=end_time-str_time
    fps=100/time_dur
    return fps


def fid_estimation(args):
    orig_file=args.test_root
    sr_file=os.path.join(args.save_folder,str(args.scale)+"x")
    orig_dataset=Dataset_LH(orig_file, transforms.Resize(299), args.scale)
    orig_len=len(orig_dataset)
    orig_dataloader=DataLoader(orig_dataset, batch_size=16, shuffle=False)
    sr_dataset = Dataset_LH(sr_file, transforms.Resize(299), args.scale)
    sr_len=len(sr_dataset)
    sr_dataloader = DataLoader(sr_dataset, batch_size=16, shuffle=False)
    fid_value = calculate_fid(orig_dataloader, sr_dataloader, False, orig_len, sr_len)
    return fid_value




def face_ver(lfw_data_loader, gen_net, fnet,args):
    features1_total = []
    features2_total = []
    labels = []
    with torch.no_grad():
        bs_total = 0
        print("calculating the identities")
        for index, (img1, img2, targets) in enumerate(tqdm(lfw_data_loader)):
            bs = len(targets)
            img2_low = low_generation(img2, args.scale)
            img2_low = 2 * (img2_low - 0.5)
            img2_low = img2_low.cuda()
            img2 = gen_net(img2_low)
            if args.wavelet_integrated:
                img2, _ = img2
            img1 = img1.cuda()
            img2 = img2.cuda()
            img1 = 2*(img1-0.5)
            img1 = img1 * 127.5 / 128
            img2 = img2 * 127.5 / 128
            img1 = img1[:, :, 9:120, 17:112]
            img2 = img2[:, :, 9:120, 17:112]
            features1 = fnet(img1).cpu()
            features2 = fnet(img2).cpu()
            features1_total += [features1]
            features2_total += [features2]
            labels += [targets]
            bs_total += bs
        #the identity features of two sets is calculated
        features1_total = torch.cat(features1_total)
        features2_total = torch.cat(features2_total)
        labels = torch.cat(labels)
        assert bs_total == 6000, print('LFW pairs should be 6,000')
        labels = labels.cpu().numpy()
        scores = nn.CosineSimilarity()(features1_total, features2_total)
        scores = scores.cpu().numpy().reshape(-1, 1)
        accuracy = []
        thd = []
        folds = KFold(n=6000, n_folds=10)
        thresholds = np.linspace(-10000, 10000, 10000 + 1)
        thresholds = thresholds / 10000
        true_positive_rate=[]
        false_positive_rate=[]
        thresholds2 = np.linspace(10000, -10000, 10000 + 1)/10000
        positive_number=labels[labels==1].sum()
        negative_number=labels.shape[0]-positive_number
        print("calculating the verification rate. please wait...")
        #true positive rate and false positive rate is calculated
        for thresh in thresholds2:
            score_mask=scores>thresh
            true_positive_number=labels[score_mask].sum()
            positive_detected_number=labels[score_mask].shape[0]
            false_positive_number=positive_detected_number-true_positive_number
            true_positive_rate.append(true_positive_number/positive_number)
            false_positive_rate.append(false_positive_number/negative_number)
        true_positive_rate_np=np.asarray(true_positive_rate)
        false_positive_rate_np=np.asarray(false_positive_rate)
        #tpr_dict = {"tpr_"+algorithm+str(scale): true_positive_rate_np }
        #fpr_dict = {"fpr_" + algorithm+str(scale): false_positive_rate_np}
        #savemat("tpr_"+algorithm+str(scale), tpr_dict)
        #savemat("fpr_" + algorithm+str(scale), fpr_dict)

    predicts = np.hstack((scores, labels))
    #finding the best threshold and verification rate in each test fold based on the rest of training folds
    for idx, (train, test) in enumerate(folds):
        best_thresh = find_best_threshold(thresholds, predicts[train])
        accuracy.append(eval_acc(best_thresh, predicts[test]))
    thd.append(best_thresh)
    mean_acc, std = np.mean(accuracy), np.std(accuracy)
    #print("mean accuracy of face verification on lfw dataset is : {0:.4f}".format(mean_acc))
    return true_positive_rate_np, false_positive_rate_np, mean_acc

