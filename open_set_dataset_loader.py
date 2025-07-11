import torch
import torch.utils.data as data_utl
from PIL import Image
import torchvision.transforms as transforms
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

class datasetLoader(data_utl.Dataset):

    def __init__(self, split_file, root, train_test, attack_type, random=True, c2i={}):
        self.class_to_id = c2i
        self.id_to_class = []

        # Class assignment
        for i in range(len(c2i.keys())):
            for k in c2i.keys():
                if c2i[k] == i:
                    self.id_to_class.append(k)
        cid = 0

        # Image pre-processing
        self.data = []
        self.transform = {
            'train':transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485],std=[0.229])
            ]),
            'test':transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485],std=[0.229])
            ]),
            'val':transforms.Compose([
            transforms.Resize([224, 224]),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485],std=[0.229])
            ])
            }

        # Reading data from CSV file
        SegInfo=[]
        with open(split_file, 'r') as f:
            for l in f.readlines():
                v= l.strip().split(',')
                if train_test == v[0]:
                    if train_test == 'train':
                        if attack_type != v[3]:
                            image_name = v[2]
                            imagePath = root +image_name
                            c = v[1]
                            attack = v[3]
                            if c not in self.class_to_id:
                                self.class_to_id[c] = cid
                                self.id_to_class.append(c)
                                cid += 1
                            # Storing data with imagepath and class
                            self.data.append([imagePath, self.class_to_id[c],attack])
                    else:
                        if attack_type == v[3] or v[3] == 'None':
                            image_name = v[2]
                            imagePath = root +image_name
                            c = v[1]
                            attack = v[3]
                            if c not in self.class_to_id:
                                self.class_to_id[c] = cid
                                self.id_to_class.append(c)
                                cid += 1
                            # Storing data with imagepath and class
                            self.data.append([imagePath, self.class_to_id[c],attack])


        self.split_file = split_file
        self.root = root
        self.random = random
        self.train_test = train_test


    def __getitem__(self, index):
        imagePath, cls ,attack_type= self.data[index]
        imageName = imagePath.split('\\')[-1]

        # Reading of the image
        path = imagePath
        img = Image.open(path)

        # Applying transformation
        tranform_img = self.transform[self.train_test](img)
        img.close()

        # Repeat NIR single channel thrice before feeding into the network
        tranform_img= tranform_img.repeat(3,1,1)

        return tranform_img[0:3,:,:], cls, imageName , attack_type

    def __len__(self):
        return len(self.data)

if __name__ == '__main__':

    dataseta = datasetLoader('../TempData/Iris_OCT_Splits_Val/test_train_split.csv', 'PathToDatasetFolder', train_test='train')

    for i in range(len(dataseta)):
        print(len(dataseta.data))
