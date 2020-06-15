import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset, TensorDataset
from torchvision import transforms, utils
import torch.nn as nn
from PIL import Image

# Parsing script arguments
parser = argparse.ArgumentParser(description='Process input')
parser.add_argument('input_folder', type=str, help='Input folder path, containing images')
args = parser.parse_args()
folder = args.input_folder

class ImageDataset(Dataset):
    def __init__(self,root,transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.idxtoImage = {i:file for i,file in enumerate(os.listdir(self.root))}

        
    def __len__(self):
        return len(self.idxtoImage)
    
    def __getitem__(self,idx):
        file = self.idxtoImage[idx]
        image = Image.open(self.root+file).convert("RGB")
        if self.transform !=None:
            image = self.transform(image)

        label = int(file.split('_')[1].split('.')[0])
        return image,label,file
    
    
transform_test = transforms.Compose([transforms.Resize((48,48)),transforms.ToTensor(),transforms.Normalize((0.5226, 0.4494, 0.4206),(0.2336, 0.2224, 0.2187))])


test_dataset = ImageDataset(folder,transform_test)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=len(test_dataset), 
                                           shuffle=False)
        
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2))
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2))
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride=2))
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=2),
            nn.ReLU())
        
        self.fc1 = nn.Linear(2*2*128, 256)
        self.ReLU = nn.ReLU()
        self.fc2 = nn.Linear(256,2)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = out.view(out.size(0), -1)
  
        out = self.fc1(out)
        out = self.ReLU(out)
        out = self.fc2(out)       

        return out    

    
    
model = Model()
model.load_state_dict(torch.load('model.pkl'))
model.eval()

y_pred = None
for images, labels,files in test_loader:
    outputs = model(images)

    _, y_pred = torch.max(outputs.data, 1)


prediction_df = pd.DataFrame(zip(files, y_pred.tolist()), columns=['id', 'label'])
####

# TODO - How to export prediction results
prediction_df.to_csv("prediction.csv", index=False, header=False)



from sklearn.metrics import f1_score
y_true = prediction_df['id'].apply(lambda x: int(x[7:8])).values
f1 = f1_score(y_true, y_pred, average='binary')		# Averaging as 'binary' - This is how we will evaluate your results.

print("F1 Score is: {:.2f}".format(f1))


