import numpy as np
import cv2
from tqdm import tqdm
import math
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import math
from torchvision import transforms
from data_collection import *
import random

device='cuda'
class SINRDataset(Dataset):
    def __init__(self):
        self.env = VIS(1)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        self.reset()
    def __len__(self):
        return 128
    def __getitem__(self, idx):
        x,y = np.random.uniform(-10,10,size=2)
        return dict(img=self.img,
                    loc=torch.FloatTensor([x,y])),self.env.get_metadata(x,y)
    def reset(self):
        self.env.n_sbs=random.randint(2,100)
        self.env.reset(hard_reset=True)
        self.img = self.normalize(torch.FloatTensor(self.env.get_image().transpose(2,0,1)))

def create_dataloader(batch_size=128):
    return DataLoader(SINRDataset(),batch_size=batch_size, shuffle=True)

model = Net().to(device)
model = nn.DataParallel(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

score = []
from tqdm import tqdm
for e in range(int(2e6)):
    e_loss = 0
    for x,y in dl:
        img = x['img'].to(device)
        loc = x['loc'].to(device)
        pred = model(img,loc).squeeze()
        loss = criterion(pred,y.to(device).float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        e_loss+=loss.item()
    score.append(e_loss)
    if not e%5:
        torch.save(model.state_dict(),f"model.pt")
    np.save("score.npy",score)
torch.save(model.state_dict(),f"model.pt")
import numpy as np
np.save("score.npy",score)