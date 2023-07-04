import torch
from omegaconf import OmegaConf
import sys
import os
from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim_ldm_ref_inpaint import DDIMSampler
import numpy as np
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
from torch.utils.data import random_split, DataLoader, Dataset, Subset
from omegaconf import OmegaConf
import configargparse
import pdb

def config_parser():
    parser = configargparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=20,)
    parser.add_argument('--numworkers', type=int, default=12,)
    parser.add_argument('--save_dir', type=str, default='./logs/inference', )

    return parser

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model

def get_model():
    config = OmegaConf.load("configs/latent-diffusion/talking-inference.yaml")
    model = load_model_from_config(config, "logs/xxx.ckpt")
    return model


parser = config_parser()
args = parser.parse_args()

model = get_model()
sampler = DDIMSampler(model)


ddim_steps = 200
ddim_eta = 0.0
use_ddim = True
log = dict()
samples = []
samples_inpainting= []
xrec_img = []


# init and save configs
config_file= 'configs/latent-diffusion/talking-inference.yaml'
config = OmegaConf.load(config_file)
data = instantiate_from_config(config.data)
dataset_configs = config.data['params']['validation']
datasets = dict([('validation', instantiate_from_config(dataset_configs))])

print("#### Data #####")
for k in datasets:
    print(f"{k}, {datasets[k].__class__.__name__}, {len(datasets[k])}")

val_dataloader = DataLoader(datasets["validation"], batch_size=args.batchsize, num_workers=args.numworkers, shuffle=False)

with torch.no_grad():
    for i, batch in enumerate(val_dataloader):
        samples = []
        samples_inpainting = []
        xrec_img = []
        z, c_audio, c_lip, c_ldm, c_mask, x, xrec, xc_audio, xc_lip = model.get_input(batch, 'image',
                                                                      return_first_stage_outputs=True,
                                                                      force_c_encode=True,
                                                                      return_original_cond=True,
                                                                      bs=args.batchsize)
        shape = (z.shape[1], z.shape[2], z.shape[3])
        N = min(x.shape[0], args.batchsize)
        c = {'audio': c_audio, 'lip': c_lip, 'ldm': c_ldm, 'mask_image': c_mask}

        b, h, w = z.shape[0], z.shape[2], z.shape[3]
        landmarks = batch["landmarks_all"]
        landmarks = landmarks / 4
        mask = batch["inference_mask"].to(model.device)
        mask = mask[:, None, ...]
        with model.ema_scope():
            samples_ddim, _ = sampler.sample(ddim_steps, N, shape, c, x0=z[:N], verbose=False, eta=ddim_eta, mask=mask)

        x_samples_ddim = model.decode_first_stage(samples_ddim)
        x_samples_ddim = torch.clamp((x_samples_ddim + 1.0) / 2.0,
                                     min=0.0, max=1.0)
        samples_inpainting.append(x_samples_ddim)

        #save images
        samples_inpainting = torch.stack(samples_inpainting, 0)
        samples_inpainting = rearrange(samples_inpainting, 'n b c h w -> (n b) c h w')
        save_path = os.path.join(args.save_dir, '105_a105_mask_face')
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        for j in range(samples_inpainting.shape[0]):
            samples_inpainting_img = 255. * rearrange(samples_inpainting[j], 'c h w -> h w c').cpu().numpy()
            img = Image.fromarray(samples_inpainting_img.astype(np.uint8))
            img.save(os.path.join(save_path, '{:04d}_{:04d}.jpg'.format(i, j)))


