import os
import argparse
import torchvision.models as models
import torch
import torch.nn as nn
import torch.optim as optim
from dataset_Loader import datasetLoader
from sophia import SophiaG 
from open_set_training import training,get_data_loaders
import timm

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

#function to load weights
def load_weights(model, weight_path):
    weights = torch.load(weight_path, map_location=device)
    # Clean keys by removing '_orig_mod.' prefix if present
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in weights['state_dict'].items()}
    model.load_state_dict(state_dict, strict=True)
    return model.to(device)

# Definition of model architecture
model1 = models.resnet50(pretrained=True)
num_ftrs = model1.fc.in_features
model1.fc = nn.Linear(num_ftrs, args.nClasses)
model1 = load_weights(model1,'ResNet50/SGD/Logs/ResNet50_best.pth')

model2 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
model2.fc = nn.Linear(2048, args.nClasses)
model2 = load_weights(model2,'DINO_ResNet50/SGD/Logs/DINO_ResNet50_best.pth')

model3 = models.densenet121(pretrained=True)
num_ftrs = model3.classifier.in_features
model3.classifier = nn.Linear(num_ftrs, args.nClasses)
model3 = load_weights(model3,'DenseNet121/SGD/Logs/DenseNet121_best.pth')

# model4 = timm.create_model('vit_base_patch16_224', pretrained=True)
# model4.head = nn.Linear(model4.head.in_features, args.nClasses)

# model5 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
# model5.fc = nn.Linear(768, args.nClasses)

#model_list = [model1, model2, model3, model4, model5]
model_list = [model1,model2,model3]
#modelNames = ['ResNet50', 'DINO_ResNet50', 'DenseNet121', 'ViT', 'DINO_ViT']
modelNames = ['ResNet50', 'DINO_ResNet50', 'DenseNet121']

# optimizers = [optim.SGD(lr=0.005, weight_decay=1e-6, momentum=0.9) , SophiaG(lr=0.005, betas=(0.965, 0.99), rho = 0.01, weight_decay=1e-3)]
# optimizerNames=['SGD', 'SophiaG']
#lr = 0.005
#lr_sched = optim.lr_scheduler.StepLR(step_size=12, gamma=0.1)

criterion = nn.CrossEntropyLoss()

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True

for i,model in enumerate(model_list):
    mName=modelNames[i]
############################################################################################
    torch.cuda.empty_cache()
    oName="SGD"
    optimizer = optim.SGD(model.parameters(),lr=0.005, weight_decay=1e-6, momentum=0.9)
    lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=12, gamma=0.1)
    training(model, dataloader, args, criterion, optimizer, lr_sched,mName,oName)
############################################################################################
    torch.cuda.empty_cache()
    # log_path = os.path.join(mName,oName, 'Logs')
    # print(os.path.join(log_path,mName+'_best.pth'))
    # weights = torch.load(os.path.join(log_path,mName+'_best.pth'),map_location='cuda')
    # new_state_dict = {}
    # for k, v in weights['state_dict'].items():
    #     if k.startswith("_orig_mod."):
    #         new_key = k[len("_orig_mod."):]
    #     else:
    #         new_key = k
    #     new_state_dict[new_key] = v
    # model.cuda()
    # model.load_state_dict(new_state_dict,strict=True)
    # optimizer = SophiaG(model.parameters(), lr=0.001, betas=(0.965, 0.99), rho = 0.02, weight_decay=1e-3)
    # lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=12, gamma=0.1)
    # oName="SophiaG"
    # training(model, dataloader, args, criterion, optimizer,lr_sched,mName,oName)
    # print('\n\n\n')






















































##EVALUATION CODE BELOW##







# import os
# import argparse
# import torchvision.models as models
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from dataset_Loader import datasetLoader
# from sophia import SophiaG 
# from new_training import training,get_data_loaders
# import timm
# from tqdm import tqdm
# from Evaluation import evaluation
# torch.cuda.empty_cache()
# torch.cuda.reset_peak_memory_stats()

# parser = argparse.ArgumentParser()
# parser.add_argument('-batchSize', type=int, default=32)
# parser.add_argument('-nEpochs', type=int, default=50)
# parser.add_argument('-csvPath', required=True, default= '../TempData/Iris_IARPA_Splits/test_train_split.csv',type=str)
# parser.add_argument('-datasetPath', required=True, default= '/PathToDatasetFolder/',type=str)
# parser.add_argument('-outputPath', required=False, default= '/OutputPath/',type=str)
# parser.add_argument('-method', default= 'ResNet50',type=str)
# parser.add_argument('-nClasses', default= 2,type=int)
# parser.add_argument('-use_amp', default= False,type=bool)

# args = parser.parse_args()
# device = torch.device('cuda')


# dataseta = datasetLoader(args.csvPath,args.datasetPath,train_test='train')
# datasetb = datasetLoader(args.csvPath,args.datasetPath, train_test='val', c2i=dataseta.class_to_id)
# datasetc = datasetLoader(args.csvPath,args.datasetPath, train_test='test', c2i=dataseta.class_to_id)

# train,val,test = get_data_loaders(dataseta, datasetb, datasetc, args.batchSize)

# dataloader = {'train': train, 'val':val, 'test':test}

# # Definition of model architecture
# model1 = models.resnet50(pretrained=True)
# num_ftrs = model1.fc.in_features
# model1.fc = nn.Linear(num_ftrs, args.nClasses)

# model2 = torch.hub.load('facebookresearch/dino:main', 'dino_resnet50')
# model2.fc = nn.Linear(2048, args.nClasses)

# model3 = models.densenet121(pretrained=True)
# num_ftrs = model3.classifier.in_features
# model3.classifier = nn.Linear(num_ftrs, args.nClasses)

# model4 = timm.create_model('vit_base_patch16_224', pretrained=True)
# model4.head = nn.Linear(model4.head.in_features, args.nClasses)

# model5 = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
# model5.fc = nn.Linear(768, args.nClasses)

# model_list = [model1, model2, model3, model4, model5]
# #model_list = [model5]
# modelNames = ['ResNet50', 'DINO_ResNet50', 'DenseNet121', 'ViT', 'DINO_ViT']
# thresh = [0.3,0.325,0.275,0.5,0.425]
# #modelNames = ['DINO_ViT']

# # optimizers = [optim.SGD(lr=0.005, weight_decay=1e-6, momentum=0.9) , SophiaG(lr=0.005, betas=(0.965, 0.99), rho = 0.01, weight_decay=1e-3)]
# # optimizerNames=['SGD', 'SophiaG']
# #lr = 0.005
# #lr_sched = optim.lr_scheduler.StepLR(step_size=12, gamma=0.1)

# criterion = nn.CrossEntropyLoss()

# torch.set_float32_matmul_precision('high')
# torch.backends.cudnn.benchmark = True

# for i,model in enumerate(model_list):
#     mName=modelNames[i]

#     weights = torch.load(os.path.join(mName,"SGD/Logs",mName+"_best.pth"),map_location='cuda')
#     new_state_dict = {}
#     for k, v in weights['state_dict'].items():
#         if k.startswith("_orig_mod."):
#             new_key = k[len("_orig_mod."):]
#         else:
#             new_key = k
#         new_state_dict[new_key] = v
#     model.cuda()
#     model.load_state_dict(new_state_dict,strict=True)
#     model.to(device)

#     print(f'Testing {mName}')
#     model.eval()
#     imgNames=[]
#     with torch.no_grad():
#         all_outputs, all_labels = [], []
#         for data, labels,imageName in tqdm(dataloader['test'], desc="Testing", leave=False):
#             data, labels = data.cuda(non_blocking=True), labels.cuda(non_blocking=True)
#             with torch.amp.autocast('cuda',enabled=args.use_amp):
#                 outputs = model(data)

#             all_outputs.append(outputs.detach().cpu())
#             all_labels.append(labels.detach().cpu())
#             imgNames.extend(imageName)

#         all_outputs = torch.cat(all_outputs, dim=0)
#         all_labels = torch.cat(all_labels, dim=0)

#     obvResult = evaluation()
#     result_path = os.path.join('new'+mName,'results')
#     os.makedirs(result_path, exist_ok=True)
#     errorIndex, predictScore, threshold = obvResult.get_result('testing', imgNames, all_labels, all_outputs, result_path,minThreshold=thresh[i])