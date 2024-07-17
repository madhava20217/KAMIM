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