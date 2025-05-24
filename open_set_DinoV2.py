import torch
import torch.nn as nn
import torch.optim as optim
from open_set_training import training,get_data_loaders
import argparse
from open_set_dataset_loader import datasetLoader

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

torch.set_float32_matmul_precision('high')
torch.backends.cudnn.benchmark = True


attacks_list = ['CL','E-display','Fake with Add On','Generated','Printed','Unknown']

for attack in attacks_list:
    print(attack)
    dataseta = datasetLoader(args.csvPath,args.datasetPath,train_test='train',attack_type=attack)
    datasetb = datasetLoader(args.csvPath,args.datasetPath, train_test='val', c2i=dataseta.class_to_id,attack_type=attack)
    datasetc = datasetLoader(args.csvPath,args.datasetPath, train_test='test', c2i=dataseta.class_to_id,attack_type=attack)

    print(f'len of train = {len(dataseta)} , test = {len(datasetc)}, val = {len(datasetb)}')
    print('\n\n')
    train,val,test = get_data_loaders(dataseta, datasetb, datasetc, args.batchSize)
    dataloader = {'train': train, 'val':val, 'test':test}

    model = DinoVisionTransformerClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0000005)
    lr_sched = optim.lr_scheduler.StepLR(optimizer,step_size=25, gamma=0.1)
    training(model, dataloader, args, criterion, optimizer, lr_sched,'DINO_with_aug',attack)

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