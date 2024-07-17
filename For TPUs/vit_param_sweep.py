# %%
import pandas as pd
import numpy as np

import torch
import torchvision
import transformers

torchvision.disable_beta_transforms_warning()

from PIL import Image

import os
import glob

import wandb
from torchinfo import summary

from utils.dataset_loaders import get_dataset, KAMIM_Dataset
from timm.scheduler.cosine_lr import CosineLRScheduler
from tqdm import tqdm

# XLA stuff
from torch_xla.amp import autocast, syncfree
import torch_xla.core.xla_model as xm
import torch_xla.distributed.parallel_loader as pl
import torch_xla.distributed.xla_multiprocessing as xmp


# utils
from utils.utils import get_pretrain_dataset_params

# KAMIM
from utils.KAMIM import ModifiedPixelShuffle, per_patch_importance

# config
from utils.config import hyperparams_pretrain as hyperparameters
from utils.config import parameters_base as parameters
from utils.config import SAVE_INTERVAL, MODEL_SAVE_INTERVAL

assert SAVE_INTERVAL%MODEL_SAVE_INTERVAL == 0       # check to ensure it doesn't crash

torch.set_num_threads(8)

# XLA config
os.environ['PJRT_DEVICE'] = 'TPU'

# TPU config 
# 4x 1 chip per process:
os.environ["TPU_CHIPS_PER_PROCESS_BOUNDS"] = "1,1,1"
os.environ["TPU_PROCESS_BOUNDS"] = "1,1,1"
# # Different per process:
os.environ["TPU_MESH_CONTROLLER_ADDRESS"] = "localhost:8476"
os.environ["TPU_MESH_CONTROLLER_PORT"] = "8476"

os.environ['WANDB_NOTEBOOK_NAME'] = "[Pretrain] ViT.ipynb"
wandb.login()

id = wandb.util.generate_id()
NUM_WORKERS = 12
PIN_MEMORY = True
accumulation_iter = 8



# parseargs
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--weight_ps',
                    type = int,
                    default = None,
                    choices = [None, 8, 16, 32],
                    help = 'The patch size for calculating keypoint density in KAMIM. Possible choices are None (for SimMIM), 8, 16, and 32.')
parser.add_argument('--temperature',
                    type = float,
                    default = None,
                    choices = [0.1, 0.25, 0.5, 1.0, 2.5],
                    help = 'The temperature value for KAMIM. Possible choices are 0.1, 0.25, 0.5, 1.0, 2.5.')


args = parser.parse_args()

KAMIM = True
weight_ps = args.weight_ps
TEMPERATURE = args.temperature


model_name = parameters['model_name']


assert weight_ps is not None
if TEMPERATURE == 0.1:
    os.environ["TPU_VISIBLE_DEVICES"] = "0" # "2,3"
    pass
elif TEMPERATURE == 0.25:
    os.environ["TPU_VISIBLE_DEVICES"] = "1" # "2,3"
    pass
elif TEMPERATURE == 1.:
    os.environ["TPU_VISIBLE_DEVICES"] = "2" # "2,3"
    pass   
elif TEMPERATURE == 2.5:
    os.environ["TPU_VISIBLE_DEVICES"] = "0" # "2,3"
    pass   
elif TEMPERATURE == 0.5:
    os.environ["TPU_VISIBLE_DEVICES"] = "3" # "2,3"
    pass   
algo = f'KAMIM [weight_ps = {weight_ps}] [T = {TEMPERATURE}]'

id

# %%
# device
DEVICE = xm.xla_device()
# %%
PROJECT_NAME = 'Param Sweep Pretrain'
RUN_NAME = f'{model_name} {algo}'

# %%
IMGNET_SRC = '../Datasets/Imagenet/train'
MODEL_SAVE_PATH = f'Models/{model_name}/KAMIM - {weight_ps} - {TEMPERATURE}/checkpoint'

if KAMIM is True:
    RUN_NAME = f'{model_name} [{algo}]'
    MODEL_SAVE_PATH = f'Models/{model_name}/KAMIM - {weight_ps} - {TEMPERATURE}/checkpoint'

# %%
DIMENSION = parameters['dimension']                                 # dimension of image
MODEL_PATCH_SIZE = parameters['model_patch_size']                   # patch size         
MASKING_PATCH_SIZE = parameters['masking_patch_size']               # masking patch size
MASK_RATIO = parameters['mask_ratio']                               # model masking ratio
INTERMEDIATE_SIZE = parameters['intermediate_size']

# model config
HIDDEN_SIZE = parameters['hidden_size']
HIDDEN_LAYERS = parameters['num_hidden_layer']
ATTN_HEADS = parameters['num_attention_heads']
                                    
NUM_PATCHES = int(DIMENSION * DIMENSION / (MASKING_PATCH_SIZE * MASKING_PATCH_SIZE)) # 192 x 192 into 32 x 32 patches
NUM_PATCHES_SIDE = int(NUM_PATCHES**(0.5))

num_masking = int(NUM_PATCHES * MASK_RATIO)

# %%
if os.path.exists(os.path.dirname(MODEL_SAVE_PATH)) == False:
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH))
    
    
print(MODEL_SAVE_PATH)

# %%
# loading pretraining dataset : ImageNet-1K
dataset_params = get_pretrain_dataset_params(DIMENSION)
dataset = get_dataset('imagenet', IMGNET_SRC, False)

pretrain_data= KAMIM_Dataset(dataset,
                             **dataset_params,
                             return_FAST=True,
                             mask_patch_size=MASKING_PATCH_SIZE,
                             model_patch_size=MODEL_PATCH_SIZE,
                             mask_ratio = MASK_RATIO)

# %% [markdown]
# ## Dataset

# %%
#hyperparameters
LR               = hyperparameters['lr']
EPOCHS           = hyperparameters['epochs']
weight_decay     = hyperparameters['weight_decay']
beta1            = hyperparameters['beta1']
beta2            = hyperparameters['beta2']
stochastic_depth = hyperparameters['stochastic_depth']

# batch size
BATCH_SIZE       = parameters['batch_size']

# warmup
warmup_epochs    = hyperparameters['warmup_epochs']    

# KAMIM
# TEMPERATURE      = hyperparameters['temperature']  # temperature for the weighting for each patch

# %%
# dataloader
pretrain_dataloader = torch.utils.data.DataLoader(pretrain_data,
                                                  batch_size = BATCH_SIZE//accumulation_iter,
                                                  shuffle = True,
                                                  num_workers= NUM_WORKERS,
                                                  pin_memory = PIN_MEMORY,
                                                  persistent_workers = True,
                                                  prefetch_factor = 2,
                                                  )

# %%
config = transformers.ViTConfig(
    hidden_size = HIDDEN_SIZE,
    num_hidden_layers = HIDDEN_LAYERS,
    num_attention_heads= ATTN_HEADS,
    image_size = DIMENSION,
    patch_size = MODEL_PATCH_SIZE,
    intermediate_size= INTERMEDIATE_SIZE,
    num_channels= 3,
)


model = transformers.ViTForMaskedImageModeling(config).train().to(DEVICE)
model.decoder[1] = ModifiedPixelShuffle(model.decoder[1].upscale_factor)

model = model.train().to(DEVICE)

# %%
# summary(model, (BATCH_SIZE//accumulation_iter, 3, DIMENSION, DIMENSION), col_names = ['input_size', 'output_size', 'num_params', 'kernel_size'])

# %%
import math

#optimiser
optim = syncfree.AdamW(model.parameters(),
                          lr = LR,
                          weight_decay = weight_decay,
                          betas = [beta1, beta2],
                          )
epoch_steps = math.ceil(len(pretrain_data)/BATCH_SIZE)
num_steps = int(EPOCHS * epoch_steps)
warmup_steps = int(warmup_epochs * epoch_steps)

lr_scheduler = CosineLRScheduler(
        optim,
        t_initial=num_steps,
        # t_mul=1.,
        lr_min=1e-5,
        warmup_lr_init=1e-6,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )

# wandB init
wandb.init(
    id = id,
    resume = 'allow',
    project = PROJECT_NAME,
    name = RUN_NAME,

    config = {

        'architecture': model_name,
        'dataset':'ImageNet1K',
        'epochs' : EPOCHS,
        'batch_size': BATCH_SIZE,
        'masking_ratio' : MASK_RATIO,
        'num_patches': NUM_PATCHES,
        'model_params': {
            'patch_size_model': MODEL_PATCH_SIZE,
            'hidden_size' : HIDDEN_SIZE,
            'num_hidden_layers' : HIDDEN_LAYERS,
            'num_attention_heads': ATTN_HEADS,
            'image_size' : DIMENSION,
            'intermediate_size': INTERMEDIATE_SIZE,
            'num_channels': 3,
        },
        'optim_params':{
            'optim': 'AdamW',
            'beta1': beta1,
            'beta2': beta2,
            'weight_decay': weight_decay,
            'learning_rate': LR,
        },
        'grad_accumulation': accumulation_iter,
        'warmup_epochs': warmup_epochs,
        'patch_size_mask' : MASKING_PATCH_SIZE,
        'KAMIM params':
            {
                'KAMIM': KAMIM,
                'temperature': TEMPERATURE,
                'Weight patch size': weight_ps,
            }
    },
)

# %%
import re

nums = [int(re.match(r'.*checkpoint_(\d+).*', x).group(1)) for x in glob.glob(MODEL_SAVE_PATH+'*[!final].pth')]

CHKPT = -1

if len(nums) != 0:
    CHKPT = max(nums)

    load_path = '{}_{}.pth'.format(MODEL_SAVE_PATH, CHKPT)
    chkpt = torch.load(load_path)

    model.load_state_dict(chkpt['model_state_dict'])
    optim.load_state_dict(chkpt['optim_state_dict'])
    # lr_scheduler.load_state_dict(chkpt['scheduler_state_dict'])
    
    print("loaded earlier settings", load_path)

# %%
LR

# %%
# training loop
model = model.train().to(DEVICE)

for epoch in range(CHKPT+1, EPOCHS+warmup_epochs):        # change back to 1
    reconst_loss = 0
    samples = 0
    
    for idx, data in (pbar := tqdm(enumerate(pretrain_dataloader), total = len(pretrain_dataloader))):

        # data
        img, feats, mask = data
        
        # append to sampels
        n_batch = len(img)
        samples+= 1/accumulation_iter

        # to device
        img = img.to(DEVICE)
        
        # feats to device
        feats = feats.to(DEVICE)
        
        # masks to device
        mask = mask.to(DEVICE)
        
        
        # get weights per patch for each img in the batch
        if KAMIM is True:
            # get weights per patch for each img in the batch
            w = per_patch_importance(feats,
                                    weight_ps,
                                    TEMPERATURE).to(DEVICE)
            w_img =  (
                w.repeat_interleave(weight_ps, 1)
                .repeat_interleave(weight_ps, 2)
                .unsqueeze(1)
                .contiguous()
                ).to(DEVICE)
            
        else:
            w_img = 1.
        
        mask_int = (
            mask.repeat_interleave(MODEL_PATCH_SIZE, 1)
            .repeat_interleave(MODEL_PATCH_SIZE, 2)
            .unsqueeze(1)
            .contiguous()
        ).to(DEVICE)

        with autocast(DEVICE):
            # forward step
            reconstruction = model.forward(img, bool_masked_pos=mask.flatten(1))['reconstruction']
        
            # derive L1 loss
            # SimMIM implementation
            loss =  torch.nn.functional.l1_loss(img*w_img, reconstruction*w_img, reduction = "none")
            loss = (loss*mask_int).sum() / (mask_int.sum() + 1e-5) /3
            loss = loss/accumulation_iter                       # grad accumulation
            
            
        # backward step
        loss.backward()
        xm.mark_step()              # graph execution to prevent OOM

        if ((idx + 1) % accumulation_iter == 0) or (idx + 1 == len(pretrain_dataloader)):
            
            #clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            
            # optimizer step
            optim.step()
            # zero grad step
            optim.zero_grad(set_to_none = True)

            # warmup
            lr_scheduler.step_update(epoch * epoch_steps + idx)

        # adding to batch loss
        reconst_loss+= loss.item()

        pbar.set_description(f"Reconstruction Loss: {reconst_loss/samples}")
        
    # warmup notification
    if epoch == (warmup_epochs-1):
        print('-'*20+'Warmup Done'+'-'*20, end = '\n\n')
    
    wandb.log({
            'epoch': epoch,
            'reconstruction_loss': reconst_loss / samples
    })

    if (epoch - warmup_epochs + 1) % MODEL_SAVE_INTERVAL == 0:
        save_path = '{}_{}.pth'.format(MODEL_SAVE_PATH, epoch)
        xm.save(
                    {
                    'epoch': epoch,
                    'model_state_dict':model.state_dict(),
                    'optim_state_dict': optim.state_dict(),
                    # 'scheduler_state_dict': lr_scheduler.state_dict(),
                    'reconstruction_loss': reconst_loss/samples,
                    'temperature': TEMPERATURE,
                    },
                save_path
                )
    if (epoch-warmup_epochs+1) % SAVE_INTERVAL == 0:
        if epoch - warmup_epochs + 1 >= 0:
            wandb.save(save_path)

# %%
save_path = "{}_final.pth".format(MODEL_SAVE_PATH)
xm.save(
    {
        'model_state_dict': model.state_dict(),
        'temperature': TEMPERATURE,
    },
    save_path
)

wandb.save(save_path)

# %%
wandb.finish()


