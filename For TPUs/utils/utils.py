import torchvision
import torch

from torchvision.transforms import v2


def get_pretrain_dataset_params(DIMENSION,
                                scale = (0.67, 1.0),
                                ratio = (3/4, 4/3),
                                hflip_p = 0.5,
                                vflip_p = 0.5,
                                antialias = False
                                ):
    resize_params = {
        'scale': scale,
        'ratio': ratio,
        'antialias': antialias
    }
    
    params = {
        'resized_crop_params': resize_params,
        'dimension': (DIMENSION, DIMENSION),
        'hflip_prob': hflip_p,
        'vflip_prob': vflip_p,
        'normalization': get_normalize()
    }
    
    return params


def get_pretrain_augment_regular(DIMENSION):
    augment = v2.Compose([
        v2.Compose([
        v2.ToImage(), 
        v2.ToDtype(torch.uint8, scale=True)
        ]),
        v2.RandomResizedCrop(size = (DIMENSION, DIMENSION), 
                                                    scale = [0.67,1], 
                                                    ratio = [3/4, 4/3],
                                                    antialias = False),
        v2.RandomVerticalFlip(),
        v2.RandomHorizontalFlip(),
        v2.ToDtype(torch.float32, scale=True)
        ])
    
    return augment

def get_rescale_normalize(DIMENSION):
    rescale_norm =  v2.Compose([
                get_rescale(DIMENSION),
                v2.Normalize(
                    mean = [0.485, 0.456, 0.406],
                    std =  [0.229, 0.224, 0.225]
                )
    ])
    return rescale_norm

def get_normalize():
    normalize = v2.Normalize(
                    mean = [0.485, 0.456, 0.406],
                    std =  [0.229, 0.224, 0.225]
                )
    return normalize

def get_rescale(DIMENSION):
    rescale =  v2.Compose([
                v2.Compose([
                    v2.ToImage(),
                    v2.ToDtype(torch.uint8, scale = True)
                ]),
                v2.Resize((DIMENSION, DIMENSION), antialias = False),
                v2.ToDtype(torch.float32, scale = True),
                ])
    return rescale

def get_cpu_augment(DIMENSION):
    augment = torchvision.transforms.Compose([
    v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale = True)
    ]),
    # v2.RandomAffine(20, (0.15, 0.15), (0.85, 1.15), 12),
    v2.Resize((DIMENSION, DIMENSION), antialias = False),
    v2.TrivialAugmentWide(),
    v2.RandomErasing(),
    v2.ToDtype(torch.float32, scale = True),
    v2.Normalize(
        mean = [0.485, 0.456, 0.406],
        std =  [0.229, 0.224, 0.225]
    )
    ])
    return augment

def batch_augment(n_classes):
    batch_augment = v2.RandomChoice([
        v2.MixUp(num_classes = n_classes),
        v2.CutMix(num_classes = n_classes)
    ])
    
    return batch_augment