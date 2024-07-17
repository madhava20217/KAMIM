SAVE_INTERVAL = 50
TEMPERATURE = 0.5
LR_DECAY = 0.65

# HYPERPARAMS TRAIN

hyperparams_train = {
    'lr': 5e-3,
    'epochs': 100,
    'weight_decay': 0.05,
    'beta1': 0.9,
    'beta2': 0.999,
    'stochastic_depth': 0.1,
    'temperature': TEMPERATURE,
    
    'warmup_epochs_paramsweep': 10,
    'warmup_epochs_linprobe': 10,
}

hyperparams_pretrain = {
    'lr': 8e-4,
    'epochs': 100,
    'weight_decay': 0.05,
    'beta1': 0.9,
    'beta2': 0.999,
    'stochastic_depth': 0.1,
    'temperature': TEMPERATURE,
    
    'warmup_epochs': 10
}


# VITs
parameters_tiny = {
    'batch_size': 2048,
    'model_name': 'ViT-T',
    'dimension' : 192,
    'model_patch_size' : 16,
    'masking_patch_size' :32,
    'mask_ratio': 0.6,
    'num_hidden_layer': 12,
    'num_attention_heads': 3,
    'hidden_size': 192,
    'intermediate_size': 768, 
    
    'batch_size_finetuning': 2048,
    'batch_size_linear_probe': 2048,

}

# VITs
parameters_custom_tinier = {
    'batch_size': 2048,
    'model_name': 'ViT-T (8L)',
    'dimension' : 192,
    'model_patch_size' : 16,
    'masking_patch_size' :32,
    'mask_ratio': 0.6,
    'num_hidden_layer': 8,
    'num_attention_heads': 3,
    'hidden_size': 192,
    'intermediate_size': 768, 
    
    'batch_size_finetuning': 2048,
    'batch_size_linear_probe': 2048,

}

# VITs
parameters_small = {
    'batch_size': 2048,
    'model_name': 'ViT-S',
    'dimension' : 192,
    'model_patch_size' : 16,
    'masking_patch_size' :32,
    'mask_ratio': 0.6,
    'num_hidden_layer': 12,
    'num_attention_heads': 6,
    'hidden_size': 384,
    'intermediate_size': 1536, 
    
    'batch_size_finetuning': 2048,
    'batch_size_linear_probe': 2048,
}

parameters_base = {
    'batch_size': 2048,
    'model_name': 'ViT-B',
    'dimension' : 192,
    'model_patch_size' : 16,
    'masking_patch_size' :32,
    'mask_ratio': 0.6,
    'num_hidden_layer': 12,
    'num_attention_heads': 12,
    'hidden_size': 768,
    'intermediate_size': 3072, 
    
    'batch_size_finetuning': 2048,
    'batch_size_linear_probe': 2048,
}

parameters_base_800ep = {
    'batch_size': 2048,
    'model_name': 'ViT-B (800ep)',
    'dimension' : 224,
    'model_patch_size' : 16,
    'masking_patch_size' :32,
    'mask_ratio': 0.6,
    'num_hidden_layer': 12,
    'num_attention_heads': 12,
    'hidden_size': 768,
    'intermediate_size': 3072, 
    
    'epochs': 800,
    'batch_size_finetuning': 2048,
    'batch_size_linear_probe': 2048,
    'warmup_epochs': 10,
}

# SWIN TRANSFORMERS
parameters_Swin_base = {
    'batch_size': 2048,
    'model_name': 'Swin-B',
    'dimension' : 192,
    'embed_dim': 128,
    'depths': [2, 2, 18, 2],
    'num_heads': [4, 8, 16, 32],
    'window_size': 6,
    'model_patch_size': 4,
    'masking_patch_size' :32,
    'mask_ratio': 0.6,
    'drop_path_rate': 0.0,
    
    'drop_path_rate_train': 0.1,
    'batch_size_finetuning': 2048,
    'batch_size_linear_probe': 2048,
}

parameters_Swin_tiny = {
    'batch_size': 2048,
    'model_name': 'Swin-T',
    'dimension' : 192,
    'embed_dim': 96,
    'depths': [2, 2, 6, 2],
    'num_heads': [3, 6, 12, 24],
    'window_size': 6,
    'model_patch_size': 4,
    'masking_patch_size' :32,
    'mask_ratio': 0.6,
    
    'drop_path_rate': 0.0,
    'drop_path_rate_train': 0.1,
    'batch_size_finetuning': 2048,
    'batch_size_linear_probe': 2048,
}