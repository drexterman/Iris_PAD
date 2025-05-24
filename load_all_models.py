import torch
import torch.nn as nn
import torchvision.models as models

import warnings
warnings.filterwarnings('ignore')

# Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
n_classes = 2  # You can make this configurable if needed

# Helper function to load weights
def load_weights(model, weight_path):
    weights = torch.load(weight_path, map_location=device)
    # Clean keys by removing '_orig_mod.' prefix if present
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in weights['state_dict'].items()}
    model.load_state_dict(state_dict, strict=True)
    return model.to(device)

# 1. Load ResNet50
resnet = models.resnet50(pretrained=True)
resnet.fc = nn.Linear(resnet.fc.in_features, n_classes)
resnet = load_weights(resnet, 'ResNet50/SGD/Logs/ResNet50_best.pth')

# 2. Load DINO ResNet50
dino_resnet = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
dino_resnet.fc = nn.Linear(2048, n_classes)
dino_resnet = load_weights(dino_resnet, 'DINO_ResNet50/SGD/Logs/DINO_ResNet50_best.pth')

# 3. Load DenseNet121
densenet = models.densenet121(pretrained=True)
densenet.classifier = nn.Linear(densenet.classifier.in_features, n_classes)
densenet = load_weights(densenet, 'DenseNet121/SGD/Logs/DenseNet121_best.pth')

# 4. Load DINOv2 ViT
dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
class DinoClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = dinov2
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, n_classes)
        )
    
    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        return self.classifier(x)

dino_vit = DinoClassifier()
dino_vit = load_weights(dino_vit, 'DINO_aug_Edisplay/Adam/Logs/DINO_aug_Edisplay_best.pth')

model_dict = {
    'resnet': resnet,
    'dino_resnet': dino_resnet,
    'densenet': densenet,
    'dinov2': dino_vit
}
threshold_dict = {
    'resnet': 0.3,
    'dino_resnet': 0.325,
    'densenet': 0.275,
    'dinov2': 0.575
}

print("All models loaded successfully!")

class EnsembleModel(nn.Module):
    def __init__(self, base_models=model_dict, thresholds = threshold_dict):
        super().__init__()
        self.base_models = nn.ModuleDict(base_models)
        self.thresholds = thresholds  # Dict of model_name: threshold
        self.num_models = len(base_models)

        # Freeze all base models
        for model in self.base_models.values():
            for param in model.parameters():
                param.requires_grad = False
            model.eval()

        # Combiner head
        self.combiner = nn.Sequential(
            nn.Linear(self.num_models, 8),
            nn.ReLU(),
            nn.Linear(8,2)
        )

    def normalize_score(self, score, threshold):
        # score: Tensor (batch_size,)
        # threshold: float
        norm = torch.where(score >= threshold,
                           0.5 + 0.5 * (score - threshold) / (1 - threshold),
                           0.5 * (score / threshold))
        return norm

    def forward(self, x):
        with torch.no_grad():
            outputs = []
            for name, model in self.base_models.items():
                raw = model(x)
                prob = torch.softmax(raw, dim=1)[:, 1]  # spoof prob
                normed = self.normalize_score(prob, self.thresholds[name])
                outputs.append(normed.unsqueeze(1))

        stacked = torch.cat(outputs, dim=1)  # shape: (batch_size, num_models)
        out = self.combiner(stacked)
        return out

# Training setup example
# ensemble = EnsembleModel(models).to(device)
# criterion = nn.BCELoss()  # Binary cross-entropy loss
# optimizer = torch.optim.Adam(ensemble.parameters(), lr=0.0001)