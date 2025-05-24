import torch
import torch.nn as nn
import torch.optim as optim
import os
from new_training import training,get_data_loaders
import argparse
from dataset_Loader import datasetLoader

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

parser = argparse.ArgumentParser()
parser.add_argument('-batchSize', type=int, default=32)
parser.add_argument('-nEpochs', type=int, default=50)
parser.add_argument('-csvPath', required=True, default= '../TempData/Iris_IARPA_Splits/test_train_split.csv',type=str)
parser.add_argument('-datasetPath', required=True, default= '/PathToDatasetFolder/',type=str)
parser.add_argument('-outputPath', required=False, default= '/OutputPath/',type=str)
parser.add_argument('-method', default='DinoV2',type=str)
parser.add_argument('-nClasses', default= 2,type=int)
parser.add_argument('-use_amp', default= False,type=bool)

args = parser.parse_args()
device = torch.device('cuda')

dataseta = datasetLoader(args.csvPath,args.datasetPath,train_test='train')
datasetb = datasetLoader(args.csvPath,args.datasetPath, train_test='val', c2i=dataseta.class_to_id)
datasetc = datasetLoader(args.csvPath,args.datasetPath, train_test='test', c2i=dataseta.class_to_id)

train,val,test = get_data_loaders(dataseta, datasetb, datasetc, args.batchSize)
dataloader = {'train': train, 'val':val, 'test':test}



dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
class DinoWithFFT(nn.Module):
    def __init__(self):
        super(DinoWithFFT, self).__init__()
        self.transformer = dinov2_vits14
        self.fft_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(224 * 224, 128),
            nn.ReLU()
        )
        self.classifier = nn.Sequential(
            nn.Linear(384 + 128, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )
    
    def forward(self, image_tensor):
        x = self.transformer(image_tensor)
        x = self.transformer.norm(x)
        fft_tensor = self.compute_fft_tensor(image_tensor)
        fft_feat = self.fft_fc(fft_tensor)
        combined = torch.cat((x, fft_feat), dim=1)
        return self.classifier(combined)

    def compute_fft_tensor(self,image_tensor):
        gray_image = image_tensor.mean(dim=1, keepdim=True)  # Convert to grayscale
        fft = torch.fft.fft2(gray_image)
        fft_shift = torch.fft.fftshift(fft)
        magnitude = torch.abs(fft_shift)
        magnitude = torch.log1p(magnitude)
        magnitude = (magnitude - magnitude.mean()) / (magnitude.std() + 1e-5)
        return magnitude  # Shape: (1, 1, H, W)


criterion = nn.CrossEntropyLoss()
torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

model = DinoWithFFT()
# weights = torch.load(os.path.join('DINO_with_aug/Adam/Logs','DINO_with_aug_best.pth'),map_location='cuda')
# new_state_dict = {}
# for k, v in weights['state_dict'].items():
#     if k.startswith("_orig_mod."):
#         new_key = k[len("_orig_mod."):]
#     else:
#         new_key = k
#     new_state_dict[new_key] = v
# model.cuda()
# model.load_state_dict(new_state_dict,strict=True)
#######################################################################################################
optimizer = optim.Adam(model.parameters(), lr=0.0000005)
lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
training(model, dataloader, args, criterion, optimizer, lr_sched,'DINO_fft_Edisplay','Adam')

#######################################################################################################
# torch.cuda.empty_cache()
# log_path = os.path.join('DINO_with_aug','Adam', 'Logs')
# weights = torch.load(os.path.join(log_path,'DINO_with_aug'+'_best.pth'),map_location='cuda')
# new_state_dict = {}
# for k, v in weights['state_dict'].items():
#     if k.startswith("_orig_mod."):
#         new_key = k[len("_orig_mod."):]
#     else:
#         new_key = k
#     new_state_dict[new_key] = v
# model.cuda()
# model.load_state_dict(new_state_dict,strict=True)
# optimizer = optim.SGD(model.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
# lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=12, gamma=0.1)
# training(model, dataloader, args, criterion, optimizer,lr_sched,'DINO_with_aug','SGD')