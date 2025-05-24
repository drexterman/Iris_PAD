import torch
import torch.nn as nn
import torchvision.models as models
from load_all_models import EnsembleModel
import argparse
from dataset_Loader import datasetLoader
from open_set_training import training,get_data_loaders

import warnings
warnings.filterwarnings('ignore')

torch.cuda.empty_cache()
torch.cuda.reset_peak_memory_stats()

parser = argparse.ArgumentParser()
parser.add_argument('-batchSize', type=int, default=32)
parser.add_argument('-nEpochs', type=int, default=50)
parser.add_argument('-csvPath', required=False, default= 'IrisPAD_2025/Combined/full_dataset.csv',type=str)
parser.add_argument('-datasetPath', required=False, default= 'IrisPAD_2025/Combined/',type=str)
parser.add_argument('-outputPath', required=False, default= '/OutputPath/',type=str)
parser.add_argument('-method', default= 'ResNet50',type=str)
parser.add_argument('-nClasses', default= 2,type=int)
parser.add_argument('-use_amp', default= False,type=bool)

args = parser.parse_args()
device = torch.device('cuda')

dataseta = datasetLoader(args.csvPath,args.datasetPath,train_test='train')
datasetb = datasetLoader(args.csvPath,args.datasetPath, train_test='val', c2i=dataseta.class_to_id)
#datasetc = datasetLoader(args.csvPath,args.datasetPath, train_test='test', c2i=dataseta.class_to_id)

train,val = get_data_loaders(dataseta, datasetb, args.batchSize)
dataloader = {'train': train, 'val':val}

model1 = EnsembleModel()

weights = torch.load("Ensemble/SGD/Logs/Ensemble_best.pth",map_location='cuda')
new_state_dict = {}
for k, v in weights['state_dict'].items():
    if k.startswith("_orig_mod."):
        new_key = k[len("_orig_mod."):]
    else:
        new_key = k
    new_state_dict[new_key] = v
model1.cuda()
model1.load_state_dict(new_state_dict,strict=True)
model1.to(device)
model1.eval()

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model1.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
lr_sched = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
training(model1, dataloader, args, criterion, optimizer, lr_sched,'Ensemble','SGD')

# model2 = EnsembleModel()
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model2.parameters(), lr=0.00001)
# lr_sched = torch.optim.lr_scheduler.StepLR(optimizer,step_size=10, gamma=0.25)
# training(model2, dataloader, args, criterion, optimizer, lr_sched,'Ensemble','Adam')