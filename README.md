# README - "Keypoint Aware Masked Image Modeling" (KAMIM)

This repository contains code for "Keypoint Aware Masked Image Modeling" (KAMIM) including code for pre-training, finetuning, and representation and attention analysis.

KAMIM attempts to exploit the differences in patches during SimMIM's reconstruction phase. This is done by using the density of keypoints extracted from handcrafted detectors like FAST (Rosten *et al.* ), SIFT (Lowe *et al.*) or ORB (Rublee *et al.*) to weight the reconstruction.

It involves two hyperparameters:

1. Patch Size ($w_{ps}$): This determines over what patch size the keypoint density will be calculated for an image. The density is calculated for each ($w_{ps} \times w_{ps}$) patch. This is followed by exponentiation and scaling to derive a weight for each patch such that the minimum possible weight is 1, corresponding to vanilla SimMIM's weighting.
2. Temperature ($T$): which determines the extent of the weighting. A higher temperature value reduces the weighting while a lower value increases it.

## Getting Started

This repository has 3 main directories:

1. Evaluations: contains the code for running the self-attention and representation analyses in the paper. We referred to the [cl-vs-mim repository](https://github.com/naver-ai/cl-vs-mim).
2. Code for GPUs: this contains the code for running SimMIM and KAMIM on GPUs. It contains the base implementation for ViTs and Swin Transformers.
3. Code for TPUs: this contains the code for running SimMIM and KAMIM on TPUs. **Note that only ViTs are supported and validated to run on TPUs**. This also contains the code for the majority of tests done -- parameter sweeps, different feature detectors, and regular pretraining using KAMIM using FAST.

### Dependencies

Each directory has a different requirements ```requirements.txt``` file available. Users can refer to that for installation.

The dataset directory is assumed to be in a ```Datasets``` directory one level above the file being run. Eg. CIFAR10 would be in a directory named ```../Datasets/CIFAR10/```. It is possible to change the directory of the dataset by changing it from within the script.

Model checkpoints may not be uniform in format -- some only contain weights and others have a dictionary organized with keys like ```model_state_dict``` and ```optim_state_dict```. In this case, the model dict can be loaded by changing the ```torch.load``` statement to load ```checkpoint['model_state_dict]```.

The following packages are required

- torch
- torchvision
- tqdm
- 

**Note: Code has been supplied for single-GPU/TPU cores. Multi-GPU/TPU training is not yet supported.**
