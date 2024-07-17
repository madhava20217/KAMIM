import torch
import torchvision
import numpy as np
import cv2
from torchvision.transforms import v2
import os


class MaskGenerator:
    def __init__(self, input_size=192, mask_patch_size=32, model_patch_size=4, mask_ratio=0.6):
        self.input_size = input_size
        self.mask_patch_size = mask_patch_size
        self.model_patch_size = model_patch_size
        self.mask_ratio = mask_ratio
        
        assert self.input_size % self.mask_patch_size == 0
        assert self.mask_patch_size % self.model_patch_size == 0
        
        self.rand_size = self.input_size // self.mask_patch_size
        self.scale = self.mask_patch_size // self.model_patch_size
        
        self.token_count = self.rand_size ** 2
        self.mask_count = int(np.ceil(self.token_count * self.mask_ratio))
        
    def __call__(self):
        mask_idx = np.random.permutation(self.token_count)[:self.mask_count]
        mask = np.zeros(self.token_count, dtype=int)
        mask[mask_idx] = 1
        
        mask = mask.reshape((self.rand_size, self.rand_size))
        mask = mask.repeat(self.scale, axis=0).repeat(self.scale, axis=1)
        
        return mask

class KAMIM_Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 base_dataset : torch.utils.data.Dataset,
                 resized_crop_params : dict = None,
                 dimension : tuple = (192, 192),
                 hflip_prob = 0.5,
                 vflip_prob = 0.5,
                 normalization = None,
                 return_FAST = True,
                 mask_patch_size = 32,
                 model_patch_size = 16,
                 mask_ratio = 0.6,
                 ):
        '''Wrapper for the pretraining dataset
        
        Arguments:
        1. base_dataset
        2. resized_crop_params: dictionary containing scale and ratio as keys
        3. dimension: the dimension of the resized image
        4. hflip_prob: probabilty to conduct horizontal flipping with
        5. vflip_prob: probability to conduct vertical flipping with
        6. normalization: post conversion normalization for base images
        7. mask_patch_size: the mask generator patch size for SimMIM
        8. model_patch_size: the patch size of the model inputs
        9. mask_ratio: the masking ratio for the patches'''
        self.base_dataset = base_dataset
        self.mask_generator = MaskGenerator(input_size = dimension[0],
                                            mask_patch_size = mask_patch_size,
                                            model_patch_size = model_patch_size,
                                            mask_ratio = mask_ratio
                                            )

        self.transforms = v2.Compose([
            v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale = True
            )]),
            v2.RandomResizedCrop(dimension, ** resized_crop_params),
            v2.RandomVerticalFlip(p = vflip_prob),
            v2.RandomHorizontalFlip(p = hflip_prob),
        ])
        self.normalize = v2.Compose([
            v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale = True
            )]),
            normalization
        ])
        self.returnFAST = return_FAST
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]
        img = self.transforms(img)
        mask = self.mask_generator()
        
        if self.returnFAST is False:     
            img = self.normalize(img)
            feat = torch.zeros_like(img)[0][None, :, :]

        else: 
            img_as_arr = img.permute(1,2,0).numpy()
            kp = cv2.FastFeatureDetector_create().detect(img_as_arr,None)
            feat = cv2.drawKeypoints(np.zeros_like(img_as_arr), kp, None, color = (255, 0, 0))[:, :, 0]
            
            feat = torch.tensor(feat)[None, :, :].sign().float()
            
            img = self.normalize(img)

        return img, feat, mask
    
    
class KAMIM_Other_Features(torch.utils.data.Dataset):
    def __init__(self,
                 base_dataset : torch.utils.data.Dataset,
                 resized_crop_params : dict = None,
                 dimension : tuple = (192, 192),
                 hflip_prob = 0.5,
                 vflip_prob = 0.5,
                 normalization = None,
                 mask_patch_size = 32,
                 model_patch_size = 16,
                 mask_ratio = 0.6,
                 feat_detector : str = 'fast',
                 ):
        '''Wrapper for the pretraining dataset
        
        Arguments:
        1. base_dataset
        2. resized_crop_params: dictionary containing scale and ratio as keys
        3. dimension: the dimension of the resized image
        4. hflip_prob: probabilty to conduct horizontal flipping with
        5. vflip_prob: probability to conduct vertical flipping with
        6. normalization: post conversion normalization for base images
        7. mask_patch_size: the mask generator patch size for SimMIM
        8. model_patch_size: the patch size of the model inputs
        9. mask_ratio: the masking ratio for the patches
        10. feat_detector: the kind of feature detector to use. Compatible types are 'fast', 'sift', 'orb' '''
        assert feat_detector in ['fast', 'sift', 'orb', 'superpoint']
        
        self.base_dataset = base_dataset
        self.mask_generator = MaskGenerator(input_size = dimension[0],
                                            mask_patch_size = mask_patch_size,
                                            model_patch_size = model_patch_size,
                                            mask_ratio = mask_ratio
                                            )

        self.transforms = v2.Compose([
            v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.uint8, scale = True
            )]),
            v2.RandomResizedCrop(dimension, ** resized_crop_params),
            v2.RandomVerticalFlip(p = vflip_prob),
            v2.RandomHorizontalFlip(p = hflip_prob),
        ])
        self.normalize = v2.Compose([
            v2.Compose([
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale = True
            )]),
            normalization
        ])
        self.detector_type = feat_detector
        
    def __len__(self):
        return len(self.base_dataset)
        
    def __getitem__(self, idx):
        img, _ = self.base_dataset[idx]
        img = self.transforms(img)
        mask = self.mask_generator()
        
        img_as_arr = img.permute(1,2,0).numpy()
        
        if self.detector_type == 'fast':
            kp = cv2.FastFeatureDetector_create().detect(img_as_arr,None)
        elif self.detector_type == 'sift':
            kp = cv2.SIFT_create().detect(img_as_arr, None)
        elif self.detector_type == 'orb':
            kp = cv2.ORB_create().detect(img_as_arr, None)
        elif self.detector_type == 'superpoint':
            # TODO implement superpoint
            pass
            
        feat = cv2.drawKeypoints(np.zeros_like(img_as_arr), kp, None, color = (255, 0, 0))[:, :, 0]
        
        feat = torch.tensor(feat)[None, :, :].sign().float()
        
        img = self.normalize(img)

        return img, feat, mask

class places365(torchvision.datasets.Places365):
    def __init__(self,
                 src,
                 small = True,
                 download = True,
                 split = 'train-standard',
                 transform = None,
                 ):
        super().__init__(src, split = split, small = small, download = download, transform = transform)
        
    def __getitem__(self, idx):
        img, tgt = super().__getitem__(idx)
        img = self.transform(img)
        return img, tgt
        
            

def get_dataset(dataset, dataset_src, train_and_test = False):
    '''Function to get the dataset and download it if required.
    
    **NOTE: ImageNet cannot be downloaded automatically. It must be present at the location specified at the 'dataset_src'
    
    Arguments:
    1. dataset: name of the dataset. Can be 'imagenet', 'cifar10', 'cifar100', 'inaturalist', and 'places365'.
    2. dataset: the source directory of the dataset.
    
    Returns: a torch.utils.data.Dataset object'''
    print(dataset_src)
    test_set = None
    if dataset == 'imagenet':
        dataset = torchvision.datasets.ImageNet(dataset_src, split = 'train')
        if train_and_test:
            test_set = torchvision.datasets.ImageNet(dataset_src, split = 'val')

    elif dataset == 'cifar10':
        os.makedirs(dataset_src, exist_ok = True)
        dataset = torchvision.datasets.CIFAR10(dataset_src,
                                               download = True,
                                               train = True)
        if train_and_test:
            test_set = torchvision.datasets.CIFAR10(dataset_src,
                                               download = True,
                                               train = False)

    elif dataset == 'cifar100':
        os.makedirs(dataset_src, exist_ok = True)
        dataset = torchvision.datasets.CIFAR100(dataset_src,
                                               download = True,
                                               train = True)
        if train_and_test:
            test_set = torchvision.datasets.CIFAR100(dataset_src,
                                               download = True,
                                               train = False)


    elif dataset == 'inaturalist':
        os.makedirs(dataset_src, exist_ok = True)
        try:
            dataset = torchvision.datasets.INaturalist(dataset_src,
                                                       version='2021_train',
                                                       download = True,
                                                       target_type = 'full')
        except:
            dataset = torchvision.datasets.INaturalist(dataset_src,
                                                       version='2021_train',
                                                       download = False,
                                                       target_type = 'full')
        if train_and_test:
            try:
                test_set = torchvision.datasets.INaturalist(dataset_src,
                                                       version='2021_valid',
                                                       download = True,
                                                       target_type = 'full')
            except:
                test_set = torchvision.datasets.INaturalist(dataset_src,
                                                       version='2021_valid',
                                                       download = False,
                                                       target_type = 'full')

    elif dataset == 'places365':
        os.makedirs(dataset_src, exist_ok = True)
        try:
            dataset = places365(dataset_src,
                                                     small = True,
                                                     download = True,
                                                     split = 'train-standard')
        except:
            dataset = places365(dataset_src,
                                                     small = True,
                                                     download = False,
                                                     split = 'train-standard')
        if train_and_test:
            try:
                test_set = places365(dataset_src,
                                                          download = True,
                                                          split = 'val',
                                                          small = True)
            except:
                test_set = places365(dataset_src,
                                                          download = False,
                                                          split = 'val',
                                                          small = True)
    else:
        raise Exception('Dataset not found')
    
    if train_and_test:
        return dataset, test_set
    
    return dataset