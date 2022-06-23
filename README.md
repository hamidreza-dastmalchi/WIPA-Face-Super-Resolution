# WIPA: Wavelet-integrated, Identity Preserving, Adversarial network for Face Super-resolution
This repository incluse the Pytorch implementation of the WIPA Face Super-Resolutio algorithm proposed in paper [Super-resolution of very low-resolution face images with a wavelet integrated, identity preserving, adversarial network](https://www.sciencedirect.com/science/article/abs/pii/S0923596522000753?dgcid=coauthor).
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
<br/>
## Datasets
The [Celebrity dataset](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) used for training the proposed FH algorithm. The database contains more than 200 K different face images under significant pose, illumination, and expression variations. In our experiment, two distinct groups of 20 thousand images are randomly selected from the CelebA dataset as our train and test dataset. In order to test the generalizing capacity of the method, we have further evaluated the performance of the proposed approach on [FW](http://vis-www.cs.umass.edu/lfw/) and [Helen dataset](http://www.ifp.illinois.edu/~vuongle2/helen/) too. All the testing and training images are roughly aligned using similarity transformation with landmarks detected by the well-known MTCNN network. The images are rescaled to the size of 128 × 128. The corresponding LR images are also constructed by down-sampling the HR images using bicubic interpolation. The experiments are accomplished in two different **scaling** factors of 8X and 16X with LR images of size 16 × 16 and 8 × 8, respectively.
