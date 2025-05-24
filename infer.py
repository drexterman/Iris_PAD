# This code is for inference using a pre-trained DINOv2 Vision Transformer model

import torch
import torch.nn as nn
from PIL import Image
import torchvision.transforms as transforms
import warnings
import argparse
warnings.filterwarnings("ignore", message="xFormers is not available")
warnings.filterwarnings("ignore", message="Can't initialize NVML")

parser = argparse.ArgumentParser()
parser.add_argument('-imagePath', type=str, default='../D-NetPAD-master/IrisPAD_2025/Combined/test/Spoof/Generated/0140_REAL_L_1_A2B.png')
#parser.add_argument('-imagePath', type=str, default='1.jpeg')
args = parser.parse_args()
#Sets device to CUDA is available, otherwise CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#defined my architecture below
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
class DinoVisionTransformerClassifier(nn.Module):
    def __init__(self):
        super(DinoVisionTransformerClassifier, self).__init__()
        self.transformer = dinov2_vits14
        self.classifier = nn.Sequential(
            nn.Linear(384, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
    
    def forward(self, x):
        x = self.transformer(x)
        x = self.transformer.norm(x)
        x = self.classifier(x)
        return x

#define model
model = DinoVisionTransformerClassifier()
#Load model , move it to GPU and set to eval mode
weights = torch.load("DINO_aug_Edisplay_best.pth",map_location='cuda')
new_state_dict = {}
for k, v in weights['state_dict'].items():
    if k.startswith("_orig_mod."):
        new_key = k[len("_orig_mod."):]
    else:
        new_key = k
    new_state_dict[new_key] = v
model.cuda()
model.load_state_dict(new_state_dict,strict=True)
model.to(device)
model.eval()

#defined transforms
Transforms = transforms.Compose([
    transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485],std=[0.229])
    ])

#here comes my image, load it , convert to rgb
image = Image.open(args.imagePath).convert('RGB')
#rotated_img = image.transpose(Image.ROTATE_90)

#apply transforms
transformed_image = Transforms(image).unsqueeze(0)

#find your label
#True = Spoof , False = Real
label = model(transformed_image.to(device))
print(label)
#label = torch.nn.functional.softmax(model(transformed_image.to(device)), dim=1)[:,1]
#True = Spoof , False = Real
# print(label.item())

# value = label.item()
# thresh = 0.575
# print(f"Confidence: {abs(value-thresh)/thresh if value <= thresh else abs(value-thresh)/(1-thresh)}, Prediction: {'Spoof' if value > thresh else 'Real'}")
