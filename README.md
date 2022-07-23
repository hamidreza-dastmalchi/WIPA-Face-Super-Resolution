# WIPA: Wavelet-integrated, Identity Preserving, Adversarial network for Face Super-resolution
Pytorch implementation of WIPA [Super-resolution of very low-resolution face images with a Wavelet Integrated, Identity Preserving, Adversarial Network]
#Paper:
(https://www.sciencedirect.com/science/article/abs/pii/S0923596522000753?dgcid=coauthor).
You can download the pre-proof version of the article [here](https://drive.google.com/file/d/1GHWiCcScPF1PK4xozoRf-88Rytom-kvl/view?usp=sharing) but  please refer to the origital manuscript for citation.
## Citation
If you find this work useful for your research, please consider citing our paper:
```
@article{DASTMALCHI2022116755,
title = {Super-resolution of very low-resolution face images with a wavelet integrated, identity preserving, adversarial network},
journal = {Signal Processing: Image Communication},
volume = {107},
pages = {116755},
year = {2022},
issn = {0923-5965},
doi = {https://doi.org/10.1016/j.image.2022.116755},
url = {https://www.sciencedirect.com/science/article/pii/S0923596522000753},
author = {Hamidreza Dastmalchi and Hassan Aghaeinia},
keywords = {Super-resolution, Wavelet prediction, Generative Adversarial Networks, Face Hallucination, Identity preserving, Perceptual quality},
```
## WIPA Algorithm
we present **Wavelet
Prediction blocks** attached to a **Baseline CNN network** to predict wavelet missing details of facial images. The
extracted wavelet coefficients are concatenated with original feature maps in different scales to recover fine
details. Unlike other wavelet-based FH methods, this algorithm exploits the wavelet-enriched feature maps as
complementary information to facilitate the hallucination task. We introduce a **wavelet prediction loss** to push
the network to generate wavelet coefficients. In addition to the wavelet-domain cost function, a combination of
**perceptual**, **adversarial**, and **identity loss** functions has been utilized to achieve low-distortion and perceptually
high-quality images while maintaining identity. The training scheme of the Wavelet-Integrated network with the combination of five loss terms is shown as below:
<p align="center">
  <img width="500" src="./block-diagram/WIPA-Training-Scheme.jpg">
</p>

## Datasets
The [Celebrity dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) used for training the proposed FH algorithm. The database contains more than 200 K different face images under significant pose, illumination, and expression variations. In our experiment, two distinct groups of 20 thousand images are randomly selected from the CelebA dataset as our train and test dataset. In order to test the generalizing capacity of the method, we have further evaluated the performance of the proposed approach on [FW](http://vis-www.cs.umass.edu/lfw/) and [Helen dataset](http://www.ifp.illinois.edu/~vuongle2/helen/) too. All the testing and training images are roughly aligned using similarity transformation with landmarks detected by the well-known MTCNN network. The images are rescaled to the size of 128 × 128. The corresponding LR images are also constructed by down-sampling the HR images using bicubic interpolation. The experiments are accomplished in two different **scaling** factors of 8X and 16X with LR images of size 16 × 16 and 8 × 8, respectively.
 **Before starting to train or test the network**, you must put the training images in the corresponding folders:
- Put training images in “.\data\train” directory.
- Put celeba test images in “.\data\test\celeba” , lfw test images in “.\data\test\lfw” and helen test images in “.\data\test\helen”.

## Pretrained Weights
The pretrained weights can be downloaded [here](https://drive.google.com/drive/folders/18V1kPDHW6F05L0xOOODNHZHO566SA6iC?usp=sharing).

## Code
The codes are consisted of two main files: the **main.py** file for training the network and the **test.py** file for evaluating the algorithm with different metrics like PSNR, SSIM and verification rate.
### Training 
To train the network, simply run this code in Anaconda terminal:
```
>>python main.py
```
We designed different input arguments for controlling the training procedure. Please use --help command to see the available input arguments. 

#### Example: 
For example, to train the wavelet-integrated network through GPU with scale factor of 8, without having pre-trained model coefficients, with learning rate of 5e-5, you can simply run the following code in the terminal:
```
python main.py –scale 8 –wi_net “” –disc_net “” –wavelet_integrated True –lr 0.00005
```

### Testing
for evaluating (testing), simply run the following code in terminal:
```
>>python test.py
```
We have also developed different options as input arguments to control the testing procedure. You can evaluate psnr, ssim, fid score and also verification rate by the “test.py” file. To do this, you have to put the test images in the corresponding folders in data root at first.

#### Example: 
For example, to evaluate the psnr and ssim of a wavelet-integrated pretrained model in scale of 8 and save the super-resolved results in folder of “./results/celeba”, you can write the following code in the command window:
```
>> test.py --wavelet_integrated True --scale 8 --wi_net gen_net_8x --save_flag True --save_folder ./results/celeba --metrics psnr ssim
```
To estimate the fid score, you have to produce the super-resolved test images first. Therefore, if you have not generated the super-resolved images, you have to call –metrics psnr ssim with fid simultaneously. You can also add the acc option to the metrics to evaluate the verification rate of the model:
```
>>python test.py --wavelet_integrated True --scale 8 --wi_net gen_net_8x --save_flag True --save_folder ./results/celeba --metrics psnr ssim fid acc
```
### Demo 
In addition, we have developed a “demo.py” python file to demonstrate the results of some sample images in the “./sample_images/gt” directory. To run the demo file, simply write the following code in terminal:
```
>>python demo.py
```
By default, the images of “./sample_images/gt” folder will be super-resolved by the wavelet-integrated network by scale factor of 8 and the results will be saved in the “./sample_images/sr” folder. To change the scaling factor, one must alter not only the –scale option but also the corresponding –wi_net argument to import the relevant pretrained state dictionary.

