import torch

class ClassificationModelViT(torch.nn.Module):
    def __init__(self, vit_backbone, hidden_layer_dim = 192, n_classes = 1000, finetuning = True, linprobe_layer = 8):
        super().__init__()
        self.vit_backbone = vit_backbone
        self.cls = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(hidden_layer_dim, n_classes)
        )
        self.finetuning = finetuning
        self.linprobe_layer = linprobe_layer
        
    

    def forward(self, input):
        if self.finetuning:
            features = self.vit_backbone(input)['last_hidden_state']
        else:
            # 8th output
            features = self.vit_backbone(input, output_hidden_states = True)['hidden_states'][self.linprobe_layer]
            
        features = features[:, 1:, :].transpose(1, 2)
        outputs = self.cls(features)
        return outputs
    
class ClassificationModelViTBatchNorm(torch.nn.Module):
    def __init__(self, vit_backbone, hidden_layer_dim = 192, n_classes = 1000, finetuning = True, linprobe_layer = 8):
        super().__init__()
        self.vit_backbone = vit_backbone
        dim = (self.vit_backbone.config.image_size//self.vit_backbone.config.patch_size)**2
        
        self.cls = torch.nn.Sequential(
            torch.nn.LayerNorm([self.vit_backbone.config.hidden_size, dim]),
            torch.nn.AdaptiveAvgPool1d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(hidden_layer_dim, n_classes)
        )
        self.finetuning = finetuning
        self.linprobe_layer = linprobe_layer
        
    

    def forward(self, input):
        if self.finetuning:
            features = self.vit_backbone(input)['last_hidden_state']
        else:
            # 8th output
            features = self.vit_backbone(input, output_hidden_states = True)['hidden_states'][self.linprobe_layer]
            
        features = features[:, 1:, :].transpose(1, 2)
        outputs = self.cls(features)
        return outputs


class ClassificationModelSwin(torch.nn.Module):
    def __init__(self, backbone, hidden_layer_dim = 768, n_classes = 1000, finetuning = True, linprobe_layer = -1):
        super().__init__()
        self.backbone = backbone
        self.cls = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool1d(output_size = 1),
            torch.nn.Flatten(),
            torch.nn.Linear(hidden_layer_dim, n_classes)
        )
        self.finetuning = finetuning
        self.linprobe_layer = linprobe_layer
        
    

    def forward(self, input):
        if self.finetuning:
            features = self.backbone(input)['last_hidden_state']
        else:
            features = self.backbone(input, output_hidden_states = True)['hidden_states'][self.linprobe_layer]
        features = features.transpose(1,2)
        outputs = self.cls(features)
        return outputs