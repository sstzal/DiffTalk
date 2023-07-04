import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import random
import cv2


class TALKBase(Dataset):
    def __init__(self,
                 txt_file,
                 data_root,
                 size=None,
                 interpolation="bicubic",
                 flip_p=0.5
                 ):
        self.data_paths = txt_file
        self.data_root = data_root
        with open(self.data_paths, "r") as f:
            self.image_paths = f.read().splitlines()
        self._length = len(self.image_paths)
        image_list_path = os.path.join(data_root, 'data.txt')
        with open(image_list_path, "r") as f:
            self.image_num = f.read().splitlines()

        self.labels = {
            "frame_id": [int(l.split('_')[0]) for l in self.image_paths],
            "image_path_": [os.path.join(self.data_root, 'images', l+'.jpg') for l in self.image_paths],
            "audio_smooth_path_": [os.path.join(self.data_root, 'audio_smooth', '105_' + l.split('_')[1] + '.npy') for l in self.image_paths],
            "landmark_path_": [os.path.join(self.data_root, 'landmarks', l+'.lms') for l in self.image_paths],
            "reference_path": [l.split('_')[0] + '_' + str(random.choice(list(set(range(1, int(self.image_num[int(l.split('_')[0])-1].split()[1])))-set(range(int(l.split('_')[1])-60, int(l.split('_')[1])+60)))))
                               for l in self.image_paths],
        }

        self.size = size
        self.interpolation = {"linear": PIL.Image.LINEAR,
                              "bilinear": PIL.Image.BILINEAR,
                              "bicubic": PIL.Image.BICUBIC,
                              "lanczos": PIL.Image.LANCZOS,
                              }[interpolation]
        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length

    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)

        image = Image.open(example["image_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        img = np.array(image).astype(np.uint8)
        image = Image.fromarray(img)
        h, w = image.size
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)
            image2 = image.resize((64, 64), resample=PIL.Image.BICUBIC)

        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)

        landmarks = np.loadtxt(example["landmark_path_"], dtype=np.float32)
        landmarks_img = landmarks[13:48]
        landmarks_img2 = landmarks[0:4]
        landmarks_img = np.concatenate((landmarks_img2, landmarks_img))
        scaler = h / self.size
        example["landmarks"] = (landmarks_img / scaler)
        example["landmarks_all"] = (landmarks / scaler)
        example["scaler"] = scaler

        #inference mask
        inference_mask = np.ones((h, w))
        points = landmarks[2:15]
        points = np.concatenate((points, landmarks[33:34])).astype('int32')
        inference_mask = cv2.fillPoly(inference_mask, pts=[points], color=(0, 0, 0))
        inference_mask = (inference_mask > 0).astype(int)
        inference_mask = Image.fromarray(inference_mask.astype(np.uint8))
        inference_mask = inference_mask.resize((64, 64), resample=self.interpolation)
        inference_mask = np.array(inference_mask)
        example["inference_mask"] = inference_mask

        #mask
        mask = np.ones((self.size, self.size))
        # zeros will be filled in
        mask[(landmarks[33][1] / scaler).astype(int):, :] = 0.
        mask = mask[..., None]
        image_mask = (image * mask).astype(np.uint8)
        example["image_mask"] = (image_mask / 127.5 - 1.0).astype(np.float32)

        example["audio_smooth"] = np.load(example["audio_smooth_path_"]) .astype(np.float32)

        reference_path = example["reference_path"].split('_')[0]
        image_r = Image.open(os.path.join(self.data_root, 'images', reference_path + '_1.jpg'))
        if not image_r.mode == "RGB":
            image_r = image_r.convert("RGB")

        img_r = np.array(image_r).astype(np.uint8)
        image_r = Image.fromarray(img_r)
        image_r = image_r.resize((self.size, self.size), resample=self.interpolation)
        image_r = np.array(image_r).astype(np.uint8)
        example["reference_img"] = (image_r / 127.5 - 1.0).astype(np.float32)

        return example


class TalkTrain(TALKBase):
    def __init__(self, **kwargs):
        super().__init__(txt_file="./data/data_train.txt", data_root="./data/HDTF", **kwargs)


class TalkValidation(TALKBase):
    def __init__(self, flip_p=0., **kwargs):
        super().__init__(txt_file="./data/data_test.txt", data_root="./data/HDTF",
                         flip_p=flip_p, **kwargs)