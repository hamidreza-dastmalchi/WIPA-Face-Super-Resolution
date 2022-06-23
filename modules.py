import torch.nn as nn
import numpy as np
import torch
import pywt
import skimage.transform as trf
seed=123
torch.manual_seed(seed)






class UpsampleBlock(nn.Module):
    def __init__(self, inpch=64, outch=64):
        super(UpsampleBlock, self).__init__()
        self.convtr = nn.ConvTranspose2d(in_channels=inpch, out_channels=outch, kernel_size=4, stride=2, padding=1, bias=False)
        self.leaky_relu=nn.LeakyReLU(0.1)

    def forward(self, x):
        out = self.leaky_relu((self.convtr(x)))
        return out




class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        leaky_relu=nn.LeakyReLU(0.1)
        self.LReLU=leaky_relu
        conv2 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        batch_norm1 = nn.BatchNorm2d(in_channels)
        batch_norm2 = nn.BatchNorm2d(in_channels)
        self.block=nn.Sequential(conv1, batch_norm1, leaky_relu, conv2, batch_norm2)

    def forward(self, x):
        out = self.LReLU(self.block(x) + x)
        return out





class GeneratorNet(nn.Module):
    def __init__(self, scale=8):
        super(GeneratorNet, self).__init__()
        self.scale=scale
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.LeakyRelu = nn.LeakyReLU(0.1)

        for i in range(2*scale-2):
            self.add_module("ResBlock" + str(i+1), ResidualBlock(64))

        for i in range(int(np.log2(scale))):
            self.add_module("Upsample" + str(i+1), UpsampleBlock(64))

        self.conv2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.tanh = nn.Tanh()
        self.upsample=nn.Upsample(128, mode="bicubic", align_corners=True)

    def forward(self, x):
        x1=self.LeakyRelu(self.bn(self.conv1(x)))
        c=1
        features=[]
        for i in range(int(np.log2(self.scale))):
            for j in range((self.scale)//(2**(i))):
                x1 = self.__getattr__("ResBlock" + str(c+j))(x1)
            x1 = self.__getattr__("Upsample" + str(i+1))(x1)
            features.append(x1)
            c+=int((self.scale)//(2**(i)))
        out0=0
        for i in range(len(features)):
            out0+=self.upsample(features[i])
        output=self.tanh(self.conv2(out0))
        return output

class GeneratorNet2(nn.Module):
    def __init__(self, scale=8):
        super(GeneratorNet2, self).__init__()
        self.scale=scale
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(64)
        self.LeakyRelu = nn.LeakyReLU(0.1)

        for i in range(2*scale-2):
            self.add_module("ResBlock" + str(i+1), ResidualBlock(64))

        for i in range(int(np.log2(scale))):
            self.add_module("Upsample" + str(i+1), UpsampleBlock(64))
            self.add_module("WaveletConv" + str(i + 1), nn.Conv2d(64,64, kernel_size=3, stride=1, padding=1, bias=False))
            self.add_module("WaveletRecConv" + str(i + 1), nn.Conv2d(64,9, kernel_size=3, stride=1, padding=1, bias=False))
            self.add_module("WaveletMergeConv" + str(i + 1), nn.Conv2d(128,64, kernel_size=1, bias=False))


        self.conv2 = nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3, stride=1, padding=1, bias=False)
        self.tanh = nn.Tanh()
        self.upsample=nn.Upsample(128, mode="bicubic", align_corners=True)

    def forward(self, x):
        x1=self.LeakyRelu(self.bn(self.conv1(x)))
        c=1
        features=[]
        XW = []
        for i in range(int(np.log2(self.scale))):
            for j in range((self.scale)//(2**(i))):
                x1 = self.__getattr__("ResBlock" + str(c+j))(x1)
            x2 = self.__getattr__("WaveletConv"+str(i+1))(x1)
            x1 = torch.cat((x1,x2),1)
            x1 = self.__getattr__("WaveletMergeConv"+str(i+1))(x1)
            xw = self.__getattr__("WaveletRecConv"+str(i+1))(x2)
            XW.append(xw)
            x1 = self.__getattr__("Upsample" + str(i+1))(x1)
            features.append(x1)
            c+=int((self.scale)//(2**(i)))
        out0=0
        for i in range(len(features)):
            out0+=self.upsample(features[i])
        output=self.tanh(self.conv2(out0))
        XW_out=[]
        for i in range(int(np.log2(self.scale))):
            XW_out.append(XW[len(XW)-1-i])

        return output, XW_out




class DiscriminatoreNet(nn.Module):
    def __init__(self):
        super(DiscriminatoreNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1)
        self.conv5 = nn.Conv2d(in_channels=512, out_channels=64, kernel_size=8, stride=1, padding=0)
        self.convtr1 = nn.ConvTranspose2d(in_channels=3, out_channels=128, kernel_size=4, stride=2, padding=1)

        self.resblock1 = ResidualBlock(in_channels=128)

        self.norm1 = nn.BatchNorm2d(64)
        self.norm2 = nn.BatchNorm2d(128)
        self.norm3 = nn.BatchNorm2d(256)
        self.norm4 = nn.BatchNorm2d(512)
        self.norm5 = nn.BatchNorm2d(128)

        self.leaky_relu = nn.LeakyReLU(0.1)
        self.block1 = nn.Sequential(self.conv1, self.leaky_relu,
                                 self.conv2, self.norm2, self.leaky_relu)
        self.block2 = nn.Sequential(self.conv3, self.norm3, self.leaky_relu,
                                 self.conv4, self.norm4, self.leaky_relu,
                                 self.conv5, self.leaky_relu)

        self.linear1=nn.Sequential(nn.Linear(in_features=64, out_features=1), nn.Sigmoid())

    def forward(self, xh):
        xh1 = self.block1(xh)
        x = xh1
        output = self.block2(x).squeeze(3).squeeze(2)
        output = self.linear1(output)
        return output






def weights_init(m):
    if (type(m)==nn.ConvTranspose2d or type(m)==nn.Conv2d):
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif (type(m)==nn.BatchNorm2d or type(m)==nn.BatchNorm1d):
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)



class VggFeatures(nn.Module):
    def __init__(self, cnn, feature_layer=11):
        super(VggFeatures, self).__init__()
        self.features = nn.Sequential(*list(cnn.features.children())[:(feature_layer+1)])

    def forward(self, x):
        return self.features(x)


def low_generation(high_image, scale):
    high_image_num = high_image.numpy().transpose(0, 2, 3, 1)
    nim, nrow, ncol, nch = high_image_num.shape
    low_image_num = np.empty((nim, nrow // scale, ncol // scale, nch), dtype=high_image_num.dtype)
    for i in range(nim):
        low_image_num[i]=trf.rescale(high_image_num[i], 1/scale, multichannel=True, anti_aliasing=True, mode="reflect")
    low_image = torch.from_numpy(low_image_num.transpose(0, 3, 1, 2))
    return low_image

def wavelet_packet(high_images, scale):
    levels = int(np.log2(scale))
    wavelet_out = []
    im_size = high_images.size()
    for i in range(levels):
        wavelet_out.append(np.empty((im_size[0], im_size[2]//(2**(i+1)),
                                     im_size[3]//(2**(i+1)), im_size[1]*3), dtype=np.float32))
    high_images_np = high_images.numpy().transpose(0,2,3,1)
    for i in range(im_size[0]):
        high_image_np_i = high_images_np[i]
        for nc in range(3):
            high_image_np_c = high_image_np_i[:,:,nc]
            wp = pywt.WaveletPacket2D(high_image_np_c, wavelet="haar", mode="symmetric", maxlevel=levels)
            for l in range(levels):
                wavelet_out_l=wavelet_out[l]
                wavelet_out_l[i,:,:,nc*3]=(wp["h"].data)/(2**l)
                wavelet_out_l[i, :, :, nc * 3+1] = (wp["v"].data)/(2**l)
                wavelet_out_l[i, :, :, nc * 3+2] = (wp["d"].data)/(2**l)
                wavelet_out[l]=wavelet_out_l
                wp = wp["a"]
    for i in range(levels):
        wavelet_out[i] = torch.from_numpy(wavelet_out[i].transpose(0,3,1,2))
    return wavelet_out




def KFold(n=6000, n_folds=10):
    folds = []
    base = list(range(n))
    for i in range(n_folds):
        test = base[int(i * n / n_folds):int((i + 1) * n / n_folds)]
        train = list(set(base) - set(test))
        folds.append([train, test])
    return folds


def eval_acc(threshold, diff):
    y_predict = np.int32(diff[:, 0] > threshold)
    y_true = np.int32(diff[:, 1])
    accuracy = 1.0 * np.count_nonzero(y_true == y_predict) / len(y_true)
    return accuracy

def find_best_threshold(thresholds, predicts):
    best_threshold = best_acc = 0
    for threshold in thresholds:
        accuracy = eval_acc(threshold, predicts)
        if accuracy >= best_acc:
            best_acc = accuracy
            best_threshold = threshold
    return best_threshold

