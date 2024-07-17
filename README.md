# README - "Keypoint Aware Masked Image Modeling" (KAMIM)

This repository contains code for "Keypoint Aware Masked Image Modeling" (KAMIM) including code for pre-training, finetuning, and representation and attention analysis.

KAMIM attempts to exploit the differences in patches during SimMIM's reconstruction phase. This is done by using the density of keypoints extracted from handcrafted detectors like FAST (Rosten *et al.* ), SIFT (Lowe *et al.*) or ORB (Rublee *et al.*) to weight the reconstruction.

It involves two hyperparameters:

1. Patch Size ($w_{ps}$): This determines over what patch size the keypoint density will be calculated for an image. The density is calculated for each ($w_{ps} \times w_{ps}$) patch. This is followed by exponentiation and scaling to derive a weight for each patch such that the minimum possible weight is 1, corresponding to vanilla SimMIM's weighting.
2. Temperature ($T$): which determines the extent of the weighting. A higher temperature value reduces the weighting while a lower value increases it.

## Getting Started

This repository has 3 main directories:

1. Evaluations: contains the code for running the self-attention and representation analyses in the paper. We referred to the [cl-vs-mim repository](https://github.com/naver-ai/cl-vs-mim).
2. Code for GPUs: this contains the code for running SimMIM and KAMIM on GPUs. It contains all the code for SimMIM and KAMIM.
3. Code for TPUs: this contains the code for running SimMIM and KAMIM on TPUs. **Note that only ViTs are supported and validated to run on TPUs**. This also contains the code for the majority of tests done -- parameter sweeps, different feature detectors, and regular pretraining using KAMIM using FAST. Please note that the models used to obtain the results in the paper were trained on TPUs.

### Dependencies

Each directory has a ```requirements.txt``` file corresponding to it. For example, GPU-based components can be installed with the ```requirements_gpu.txt```  file. For TPUs, the ```requirements.txt``` file is different and can be found in the ```For TPUs/``` directory.

The dataset directory is assumed to be in a ```Datasets``` directory one level above the file being run. Eg. CIFAR10 would be in a directory named ```../Datasets/CIFAR10/```. It is possible to change the directory of the dataset by changing it from within the script.

Model checkpoints may not be uniform in format -- some only contain weights and others have a dictionary organized with keys like ```model_state_dict``` and ```optim_state_dict```. In this case, the model dict can be loaded by changing the ```torch.load``` statement to load ```checkpoint['model_state_dict]```.

The following packages are required for running the GPU-based training scripts:

- torch>=2.2
- torchvision>=0.17
- pandas==2.2.2
- numpy==1.26.4
- transformers>=4.38.1
- wandb>=0.16.3
- torchinfo==1.8.0
- torchmetrics>=1.3.1
- tqdm>=4.64.1
- matplotlib>=3.8.3
- seaborn==0.13.2
- scikit-learn==1.5.0
- opencv-python==4.9.0.80
- opencv-python-headless==4.9.0.80
- opencv-contrib-python==4.9.0.80
- timm>=0.9.16


For the Evaluations portion, the following are also required:
- fastai>=2.7.15
- torch-fidelity>=0.3.0
- **transformers==4.38.1** (strictness different from the GPU-training portion)
- **timm==0.5.4** (the version differs from the GPU-training portion).
- einops==0.6.0

**Note: Code has been supplied for single-GPU/TPU cores. Multi-GPU/TPU training is not yet supported.**

## Running code

### GPU-based training

Go to the ```For GPUs``` directory. Install dependencies using ```pip install -r requirements.txt```.

For training ViTs, please use the ```vit.py``` script. Similarly, for training Swin transformers, please use the ```swin.py``` script. For finetuning/linear probing with ViTs, please use ```finetune_vit.py```. Similarly, use ```finetune_swin.py``` for Swin transformers.

#### Pretraining

The arguments to the ```vit.py``` and the ```swin.py``` scripts are:
1. model: specifies which model to use. For ViTs, the allowed options are: 'vit_t', 'vit_s', and 'vit_b'. For Swin transformers, the allowed options are: 'swin_t'and swin_b'.
2. KAMIM: a flag that forces pretraining by KAMIM. This then requires the next two 'weight_ps' and 'temperature' parameters to be set.
3. weight_ps: integer specifying the keypoint density calculation patch size.
4. temperature: float specifying the temperature to be used with KAMIM.
5. dataset: the dataset to pretrain/finetune on. 
6. device: the device to use for torch.
7. accumulation: the gradient accumulation steps to be used.
8. save_interval: the rate at which the model should be pushed to wandb.
9. model_save_interval: the rate at which the model should be saved to local disk.

Example command for training a ViT-B with KAMIM with $w_{ps} = 8, T = 0.25$ on CIFAR10 on device 0 with 8 steps of gradient accumulation, and model save interval and save intervals of 25 epochs.

```
python3 vit.py \
	--model=vit_b \
	--KAMIM \
	--weight_ps=8 \
	--temperature=0.25 \
	--dataset=cifar10 \
	--device=0 \
	--accumulation=8 \
	--save_interval=25 \
	--model_save_interval=25
```

For Swin-B:

```
python3 swin.py \
	--model=swin_b \
    --detector=fast \
	--KAMIM \
	--weight_ps=8 \
	--temperature=0.25 \
	--dataset=cifar10 \
	--device=0 \
	--accumulation=8 \
	--save_interval=25 \
	--model_save_interval=25
```

For finetuning, please use the following:

```
python3 finetune_vit.py \
	--model=vit_b \
    --detector=fast \
	--linear_probing \
	--KAMIM \
	--weight_ps=8 \
	--temperature=0.25 \
	--dataset=cifar10 \
	--device=0 \
	--accumulation=8 \
	--save_interval=25 \
	--model_save_interval=25 \
	--use_avgpooling=False \
	--linprobe_layer=8
```

For Swin transformers:

```
python3 finetune_swin.py \
	--model=swin_b \
    --detector=fast \
	--linear_probing \
	--KAMIM \
	--weight_ps=8 \
	--temperature=0.25 \
	--dataset=cifar10 \
	--device=0 \
	--accumulation=8 \
	--save_interval=25 \
	--model_save_interval=25 \
```

By default, Swin transformers use the last layer for linear probing, which already uses layernorm.


The number of epochs, warmup epochs, learning rate, and values of $\beta_1$ and $beta_2$ for AdamW are held constant and available in ```utils/config.py```.

The finetuning scripts require the ```checkpoint_final.pth``` file for all models. It is possible to rename earlier checkpoints to this for this purpose.

#### Performance and Checkpoints

| Model  | SimMIM (LP) | KAMIM (LP) | SimMIM (FT) | KAMIM (FT) | #Params | SimMIM Checkpoint | KAMIM Checkpoint (FAST) |
|--------|-------------|------------|-------------|------------|---------|-------------------|------------------|
| ViT-T  | 12.37       | **13.75**  | **70.49**   | 70.41      | 5.5M    | [imagenet_chkpt](https://drive.google.com/file/d/1OUzHjR3G2hddBpn5OOXf1eOVhBShEhj2/view?usp=drive_link)    | [imagenet_chkpt](https://drive.google.com/file/d/1vavaCctCCtYtUnPAyB6tPE5sLn6MdswR/view?usp=drive_link)   |
| ViT-S  | 20.99       | **22.68**  | 76.8        | **77.02**  | 21.6M   | [imagenet_chkpt](https://drive.google.com/file/d/1E_QQrqYJizxjlYrO-qspoiTL9mm6z1G7/view?usp=drive_link)    | [imagenet_chkpt](https://drive.google.com/file/d/1ukDoyR2DX8M2k5icDy_eG8MLHbMsA7-h/view?usp=drive_link)   |
| ViT-B  | 16.12       | **33.97**  | 76.78       | **77.30**  | 85.7M   | [imagenet_chkpt](https://drive.google.com/file/d/1ThI6gBQxKWLE940btmVLVkoZmz5CSycj/view?usp=drive_link)    | [imagenet_chkpt](https://drive.google.com/file/d/1l69YLqO2hrQBTt6GY4yPbr4KfLU2H5un/view?usp=drive_link)   |
| Swin-T | 14.37       | **14.53**  | 77.94       | **78.12**  | 27.5M   | [imagenet_chkpt](https://drive.google.com/file/d/1F0MlOnvNoq6KTmx9B4vA8OQZWxoU4Gsy/view?usp=drive_link)    | [imagenet_chkpt](https://drive.google.com/file/d/1kjhk2wOF9aTO7IHQU_NdmbK41NZ6ZQYl/view?usp=drive_link)   |
| Swin-B | 20.42       | **18.16**  | 79.58       | **80.02**  | 86.     | [imagenet_chkpt](https://drive.google.com/file/d/1Bwhs0VLRmqJk9DsM01J5Nm5kpNntvOV-/view?usp=drive_link)    | [imagenet_chkpt](https://drive.google.com/file/d/1mjN-6KGUmTEV_P2LrGvtrYlR52UD9zgH/view?usp=drive_link)   |

**Note:** please keep the pretrained checkpoints for SimMIM at: ```<main_dir>/Models/<dataset>/<model>/<algorithm>/<detector - only if KAMIM>/checkpoint_final.pth```. Eg, for GPU-based runs on Imagenet with a ViT-B with $w_{ps} = 8, T = 0.25$ and with the ORB detector, the path would be ```For GPUs/Models/imagenet/ViT-B/KAMIM - 8 - 0.25/orb/checkpoint_final.pth```. Similarly, for SimMIM, it would be ```For GPUs/Models/imagenet/ViT-B/SimMIM/checkpoint_final.pth```.

**Note 2:** The provided checkpoints have been trained for 100 epochs with 10 epochs of warmup with a Cosine LR scheduler and an AdamW optimizer with LR = 1e-4 (please refer to the paper for details). FAST keypoints are used with KAMIM for these checkpoints.


### Representation Analysis

#### Prerequisites
There are two sections in this directory: performance on reconstructing images, and analysing the representations and self-attentions.

In both of these directories, a ```checkpoints``` directory should be made and checkpoints with KAMIM and SimMIM be kept as ```kamim_checkpoint_final.pth``` and ```simmim_checkpoint_final.pth``` respectively.

In addition to this, the Imagenet dataset must be available to run the ```compare_reconstructions.ipynb```. The path details can be found at the first cell in the notebook.

#### Running the code

1. Reconstruction comparisons: notebook at ```Performance on Reconstruction/compare_reconstructions.ipynb```. Gives examples of reconstructed images and P-SNR and SSIM scores.
2. Representation Analysis: notebook at ```Representation Analysis/representation_analysis.ipynb```. Gives T-SNE plots and fourier transform analysis of tokens, and extent of transformation by self-attention on tokens.
3. Self-attention analysis: notebook at ```Representation Analysis/self-attention_analysis.ipynb```. Gives plots of attention maps based on query, attention distance, and NMI.


###  TPU-based training

Execute commands from ```setup_tpu.sh``` to setup the environment. The requirements.txt file is given within the ```For TPUs``` directory, please use that.

The code is similar to the GPU-based training part. The provided checkpoints work for TPUs as well. There may be some flags missing when executing files except the base ```vit.py``` and ```finetune_vit.py``` scripts.

#### Description of files

1. ```vit.py``` and ```finetune_vit.py```: Base ViT pretraining and finetuning. The model architecture will need to be changed from within the script. There is no command line argument for that, unlike the GPU-based scripts.
2. ```vit_param_sweep.py``` and ```finetune_paramsweep_Vit.py```: for sweeping over all hyperparameters tested in the paper with a ViT-B. The model architecture, feature extractor, algorithm (KAMIM) is fixed in this case.
3. ```vit_feat_detector_diff.py``` and ```finetune_vit_diff_feat_detector.py```: tests with different feature detectors like SIFT and ORB.