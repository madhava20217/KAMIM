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
import re

import wandb
from torchinfo import summary
import torchmetrics
from timm.scheduler.cosine_lr import CosineLRScheduler

from tqdm import tqdm

import math

# utils
from utils.utils import get_cpu_augment, batch_augment, get_rescale_normalize
from utils.dataset_loaders import get_dataset

# config
from utils.config import hyperparams_train as hyperparameters
from utils.config import ARCHITECTURES                  
from utils.config import LR_DECAY

# eval helper
from utils.eval_helpers import ClassificationModelViT as ClassificationModel

torch.set_num_threads(24)

wandb.login()

id = wandb.util.generate_id()
NUM_WORKERS = 24
PIN_MEMORY = True
# parseargs
import argparse
parser = argparse.ArgumentParser()

parser.add_argument('--model',
                    type = str,
                    default = 'vit_b',
                    help = "Model to train. Will load the config from utils.config.py. For ViT-T, use 'vit_t', for ViT-S, use 'vit-s', and for ViT-B, use 'vit_b'. In order to change parameters, please modify the code there.")
parser.add_argument('--detector',
                    type = str,
                    default = None,
                    choices = [None, 'fast', 'orb', 'sift'],
                    help = "The detector to use for obtaining keypoints. Valid options are 'fast', 'orb' and 'sift'. For SimMIM, use 'None'.")
parser.add_argument("--linear_probing",
                    action = 'store_false',
                    help = 'Add this flag if you want to conduct linear probing. Absence of this flag will make the training default to finetuning.')
parser.add_argument("--KAMIM",
                    action = 'store_true',
                    help = 'The pretraining style to use. Add this flag for using KAMIM.')
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
parser.add_argument("--dataset",
                    type = str,
                    default = 'imagenet',
                    choices = ['imagenet', 'cifar10', 'cifar100', 'inaturalist', 'places365'],
                    help = "The dataset to train on. Currently supportes one of 'imagenet', 'cifar10', 'cifar100', 'inaturalist', and 'places365'.")
parser.add_argument('--device',
                    type = str,
                    default = '0',
                    help = 'The device to use for training. Defaults to 0.')
parser.add_argument('--accumulation',
                    type = int,
                    default = 2,
                    help = 'The gradient accumulation size. Note: The batch size for all tests is 2048 editable from the utils.config.py file.')
parser.add_argument('--save_interval',
                    type = int,
                    default = 100,
                    help = 'Interval (in #epochs) for pushing model checkpoints to wandb. Defaults to 100.')
parser.add_argument('--model_save_interval',
                    type = int,
                    default = 100,
                    help = 'Interval (in #epochs) for saving model checkpoints. Defaults to 100.')
parser.add_argument('--use_avgpooling',
                    type = bool,
                    default = False,
                    help = 'Whether to use average pooling when conducting linear probing. Add this flag if you want to use average pooling.')
parser.add_argument('--linprobe_layer',
                    type = int,
                    default = 8,
                    help = 'The layer to use for conducting linear probing. Defaults to 8.')

args = parser.parse_args()

cfg = args.model
parameters = ARCHITECTURES[cfg]

model_name = parameters['model_name']

kp_detector = args.detector
linprobe_layer = args.linprobe_layer
finetuning = args.linear_probing
KAMIM = args.KAMIM
weight_ps = args.weight_ps
TEMPERATURE = args.temperature          # temperature, only used for KAMIM
dataset = args.dataset                  # pretraining dataset
accumulation_iter = args.accumulation   # gradient accumulation
use_avgpool = args.use_avgpooling       # to use average pooling for features

SAVE_INTERVAL, MODEL_SAVE_INTERVAL = args.save_interval, args.model_save_interval
assert SAVE_INTERVAL%MODEL_SAVE_INTERVAL == 0

if finetuning is True:
    process = 'Finetuning'
else:
    process = 'Linear Probing'
    
if KAMIM is True:
    assert weight_ps is not None
    algo = 'KAMIM'
else:
    algo = 'SimMIM'
    
print(id)

# %%
device = f'cuda:{args.device}'
DEVICE = torch.device(device)

# %%
PROJECT_NAME = f'KAMIM ({process})'
RUN_NAME = f'[{dataset}] [{kp_detector}] {model_name} [{algo}]'
MODEL_SAVE_PATH = f'Benchmarks/{dataset}/{model_name}/{process}/{algo}/checkpoint'
WEIGHT_PATH = f'Models/{dataset}/{model_name}/{algo}/checkpoint_final.pth'

if KAMIM is True:
    WEIGHT_PATH = f'Models/{dataset}/{model_name}/{algo} - {weight_ps} - {TEMPERATURE}/{kp_detector}/checkpoint_final.pth'      # IN weight
    RUN_NAME = f'[{dataset}] [{kp_detector}] {model_name} [{algo}] [WP_Size = {weight_ps}] [T = {TEMPERATURE}]'
    MODEL_SAVE_PATH = f'Benchmarks/{dataset}/{model_name}/{process}/{algo} - {weight_ps} - {TEMPERATURE}/{kp_detector}/checkpoint'
print(WEIGHT_PATH)

# %%
if dataset == 'imagenet':
    dataset_src = '../Datasets/Imagenet/'
elif dataset == 'cifar10':
    dataset_src = '../Datasets/CIFAR10/'
elif dataset == 'cifar100':
    dataset_src = '../Datasets/CIFAR100/'
elif dataset == 'inaturalist':
    dataset_src = '../Datasets/iNaturalist/'
elif dataset == 'places365':
    dataset_src = '../Datasets/Places365/'
# %%
if not os.path.exists(os.path.dirname(MODEL_SAVE_PATH)):
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

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

#hyperparameters
LR               = hyperparameters['lr']
EPOCHS           = hyperparameters['epochs']
weight_decay     = hyperparameters['weight_decay']
beta1            = hyperparameters['beta1']
beta2            = hyperparameters['beta2']
stochastic_depth = hyperparameters['stochastic_depth']

# batch size
# warmup
if finetuning:
    BATCH_SIZE   = parameters['batch_size_finetuning']
    warmup_epochs= hyperparameters['warmup_epochs_paramsweep']   
else:
    BATCH_SIZE   = parameters['batch_size_linear_probe']
    warmup_epochs= hyperparameters['warmup_epochs_linprobe']


# defining augmentation as per SimMIM
rescale_norm = get_rescale_normalize(DIMENSION)

augment = get_cpu_augment(DIMENSION)


training_set, validation_set = get_dataset(dataset,
                                        dataset_src,
                                        train_and_test = True)

training_set.transform = augment
validation_set.transform = rescale_norm

try:
    NUM_CLASSES = len(training_set.classes)
except:
    if dataset == 'inaturalist':
        NUM_CLASSES = len(training_set.all_categories)
batch_augment = batch_augment(NUM_CLASSES)


# %%
# dataloader
train_dataloader = torch.utils.data.DataLoader(training_set,
                                                  batch_size = BATCH_SIZE//accumulation_iter,
                                                  shuffle = True,
                                                  num_workers= NUM_WORKERS,
                                                  pin_memory = PIN_MEMORY,
                                                  prefetch_factor = 2,
                                                  persistent_workers = True)

val_dataloader = torch.utils.data.DataLoader(validation_set,
                                             batch_size = BATCH_SIZE//accumulation_iter,
                                             shuffle = False,
                                             num_workers = NUM_WORKERS,
                                             pin_memory = PIN_MEMORY,
                                             prefetch_factor = 2,
                                             persistent_workers = True)

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


backbone = transformers.ViTForMaskedImageModeling(config)

final_checkpoint = torch.load(WEIGHT_PATH,
                              map_location = {
                                  'cuda:1': 'cpu', 
                                  'cuda:0': 'cpu'
                                  }
                              )
if type(final_checkpoint) == dict and 'model_state_dict' in final_checkpoint.keys():
    final_checkpoint = final_checkpoint['model_state_dict']
model_state_dict = final_checkpoint

backbone.load_state_dict(model_state_dict)

# %%
backbone = backbone.base_model
model = ClassificationModel(backbone,
                            hidden_layer_dim = HIDDEN_SIZE,
                            n_classes = NUM_CLASSES,
                            finetuning = finetuning,
                            linprobe_layer = linprobe_layer,
                            use_avgpool = use_avgpool).to(DEVICE)

# %%
# summary(model, (BATCH_SIZE//accumulation_iter, 3, DIMENSION, DIMENSION), col_names = ['input_size', 'output_size', 'num_params', 'kernel_size'])

# %%
# linear probing
import math

if finetuning is False:
    for param in model.vit_backbone.parameters():
        param.requires_grad = False

#optimiser
optim = torch.optim.AdamW(model.parameters(),
                          lr = LR,
                          weight_decay = weight_decay,
                          betas = [beta1, beta2],
                          )

epoch_steps = math.ceil(len(training_set)/BATCH_SIZE)
num_steps = int(EPOCHS * epoch_steps)
warmup_steps = int(warmup_epochs * epoch_steps)

lr_scheduler = CosineLRScheduler(
        optim,
        t_initial=num_steps,
        # t_mul=1.,
        lr_min=2.5e-6,
        warmup_lr_init=2.5e-6,
        warmup_t=warmup_steps,
        cycle_limit=1,
        t_in_epochs=False,
    )

# wandB init
wandb.init(
    id = id,# id,
    resume =  'allow',
    project = PROJECT_NAME,
    name = RUN_NAME,

    config = {
        'architecture': model_name,
        'dataset': dataset,
        'epochs' : EPOCHS,
        'batch_size': BATCH_SIZE,
        'masking_ratio' : MASK_RATIO,
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
        'accumulation_iters': accumulation_iter,
        'patch_size_mask' : MASKING_PATCH_SIZE,
        'KAMIM settings':  {
                'KAMIM': KAMIM,
                'temperature': TEMPERATURE,
                'Weight patch size': weight_ps,
                'Detector': kp_detector,
            }, 
        'linprobe_layer': linprobe_layer,
        # 'LR_decay': LR_DECAY
    },
)

# %%
nums = [int(re.match(r'.*checkpoint_(\d+).*', x).group(1)) for x in glob.glob(MODEL_SAVE_PATH+'*[!final].pth')]

CHKPT = -1

if len(nums) != 0:
    CHKPT = max(nums)

    load_path = '{}_{}.pth'.format(MODEL_SAVE_PATH, CHKPT)
    chkpt = torch.load(load_path, map_location = {'cuda:1': 'cpu', 
                                                  'cuda:0': 'cpu'})

    model.load_state_dict(chkpt['model_state_dict'])
    optim.load_state_dict(chkpt['optim_state_dict'])
    # lr_scheduler.load_state_dict(chkpt['scheduler_state_dict'])
    
    print(load_path)
    
    print("loaded earlier settings")

# %%
# training loop
BCELoss = torch.nn.CrossEntropyLoss().to(DEVICE)
model = model.train().to(DEVICE)
scaler = torch.cuda.amp.grad_scaler.GradScaler()

val_acc_top1 = torchmetrics.Accuracy(task = 'multiclass',
                                  num_classes = NUM_CLASSES,
                                  top_k = 1).to(DEVICE)
val_acc_top5 = torchmetrics.Accuracy(task = 'multiclass',
                                  num_classes = NUM_CLASSES,
                                  top_k = 5).to(DEVICE)

for epoch in range(CHKPT+1, EPOCHS+warmup_epochs):        # change back to 1
    train_loss = 0
    samples = 0
    val_acc_top1.reset()
    val_acc_top5.reset()
    model.train()
    
    for idx, data in (pbar := tqdm(enumerate(train_dataloader), total = len(train_dataloader))):

        # data
        img, cls = data
        
        # append to samples
        n_batch = len(img)
        samples+= 1/accumulation_iter

        with torch.no_grad():
            img, cls = batch_augment(img, cls)                                                      # batch augment
        
        # to device
        img = img.to(DEVICE)
        cls = cls.to(DEVICE)
        

        with torch.autocast(device_type = 'cuda', dtype = torch.bfloat16):
        # forward step
            output = model.forward(img)
            # derive BCELoss
            loss = BCELoss(output, cls)
            loss = loss/accumulation_iter                                                           # GRAD ACCUMULATION

        # backward step
        scaler.scale(loss).backward()
        
        if ((idx + 1) % accumulation_iter == 0) or (idx + 1 == len(train_dataloader)):
            
            #clipping
            scaler.unscale_(optim)
            # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            
            # optimizer step
            scaler.step(optim)
            scaler.update()
            # zero grad step
            optim.zero_grad(set_to_none = True)

            # warmup
            lr_scheduler.step_update(epoch * epoch_steps + idx)
        


        # adding to batch loss
        train_loss+= (loss.item())

        pbar.set_description(f"Training Loss: {train_loss/samples}")
    
    val_loss = 0
    val_samples = 0
    model.eval()
    # validation
    for val_data in val_dataloader:
        
        with torch.no_grad():
            val_img, val_cls = val_data
            val_img = val_img.to(DEVICE)
            val_cls = val_cls.to(DEVICE)
            
            n_batch_val = len(val_img)
            val_samples+= 1
            
            val_output = model.forward(val_img)
            batch_val_loss = BCELoss(val_output, val_cls)
            
            val_loss+= (batch_val_loss.item())
            val_acc_top1(val_output, val_cls)
            val_acc_top5(val_output, val_cls)    
            
    if epoch == (warmup_epochs-1):
        print('-'*20+'Warmup Done'+'-'*20, end = '\n\n')
        
    wandb.log({
            'epoch': epoch,
            'train_loss': train_loss / samples,
            'val_loss': val_loss/val_samples,
            'val_acc_top1': val_acc_top1.compute().item(),
            'val_acc_top5': val_acc_top5.compute().item()
    })


    if (epoch - warmup_epochs + 1) % MODEL_SAVE_INTERVAL == 0:
        save_path = '{}_{}.pth'.format(MODEL_SAVE_PATH, epoch)
        torch.save(
                    {
                    'epoch': epoch,
                    'model_state_dict':model.state_dict(),
                    'optim_state_dict': optim.state_dict(),
                    'dataset': dataset,
                    'detector': kp_detector,
                    'train_loss': train_loss / samples,
                    'val_loss': val_loss/val_samples,
                    'val_acc_top1': val_acc_top1.compute().item(),
                    'val_acc_top5': val_acc_top5.compute().item()
                    },
                save_path
                )
    if (epoch-warmup_epochs+1) % SAVE_INTERVAL == 0 and (epoch + 1 != warmup_epochs):
        if epoch - warmup_epochs + 1 >= 0:
            wandb.save(save_path)

# %%
save_path = "{}_final.pth".format(MODEL_SAVE_PATH)
torch.save(
    model.state_dict(),
    save_path
)

wandb.save(save_path)

# %%
wandb.finish()


