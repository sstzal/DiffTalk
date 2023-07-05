# DiffTalk #
The pytorch implementation for our CVPR2023 paper "DiffTalk: Crafting Diffusion Models for Generalized Audio-Driven Portraits Animation".

[[Project]](https://sstzal.github.io/DiffTalk/) [[Paper]](https://openaccess.thecvf.com/content/CVPR2023/papers/Shen_DiffTalk_Crafting_Diffusion_Models_for_Generalized_Audio-Driven_Portraits_Animation_CVPR_2023_paper.pdf) [[Video Demo]](https://cloud.tsinghua.edu.cn/f/e13f5aad2f4c4f898ae7/)

## Requirements
- python 3.7.0
- pytorch 1.10.0
- pytorch-lightning 1.2.5
- torchvision 0.11.0
- pytorch-lightning==1.2.5

For more details, please refer to the `requirements.txt`. We conduct the experiments with 8 NVIDIA 3090Ti GPUs.

Put the [pre-trained model](https://cloud.tsinghua.edu.cn/f/7eb11fc208144ed0ad20/?dl=1) for the first stage to `./models`.

## Dataset
Please download the HDTF dataset for training and test, and process the dataset as following.

**Data Preprocessing:** 


1. Set all videos to 25 fps.
2. Extract the audio signals and facial landmarks. 
3. Put the processed data in `./data/HDTF`, and construct the data directory as following.
4. Constract the `data_train.txt` and `data_test.txt` as following.

./data/HDTF:

    |——data/HDTF
       |——images
          |——0_0.jpg
          |——0_1.jpg
          |——...
          |——N_M.bin
       |——landmarks
          |——0_0.lmd
          |——0_1.lmd
          |——...
          |——N_M.lms
       |——audio_smooth
          |——0_0.npy
          |——0_1.npy
          |——...
          |——N_M.npy

./data/data_train(test).txt:

    0_0
    0_1
    0_2
    ...
    N_M


N is the total number of classes, and M is the class size.


## Training
```
sh run.sh
```

## Test
```
sh inference.sh
```
## Weakness
1. The DiffTalk models talking head generation as an iterative denoising process, which needs more time to synthesize a frame compared with most GAN-based approaches. This is also a common problem of LDM-based works.
2. The model is trained on the HDTF dataset, and it sometimes fails on some identities from other datasets. 
3. When driving a portrait with more challenging cross-identity audio, the audio-lip synchronization of the synthesized video is slightly inferior to the ones under self-driven setting.
4. During inference, the network is also sensitive to the mask shape in z_T , where the mask needs to cover the mouth region completely and its shape cannot leak any
lip shape information.

## Acknowledgement 
This code is built upon the publicly available code [latent-diffusion](https://github.com/CompVis/latent-diffusion). Thanks the authors of latent-diffusion for making their excellent work and codes publicly available. 

## Citation ##
Please cite the following paper if you use this repository in your research.

```
@inproceedings{shen2023difftalk,
   author={Shen, Shuai and Zhao, Wenliang and Meng, Zibin and Li, Wanhua and Zhu, Zheng and Zhou, Jie and Lu, Jiwen},
   title={DiffTalk: Crafting Diffusion Models for Generalized Audio-Driven Portraits Animation},
   booktitle={CVPR},
   year={2023}
}
```
