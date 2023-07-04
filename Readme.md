# DiffTalk #
The pytorch implementation for our submitted paper "DiffTalk: Crafting Diffusion Models for Generalized Audio-Driven Portraits Animation".

## Requirements
- python 3.7.0
- pytorch 1.10.0
- pytorch-lightning 1.2.5
- torchvision 0.11.0
- pytorch-lightning==1.2.5

For more details, please refer to the `requirements.txt`. We conduct the experiments with 8 NVIDIA 3090Ti GPUs.

Put the pre-trained model for the first stage to `./models`.

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
