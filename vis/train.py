import numpy as np
import cv2
from tqdm import tqdm
import math
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch
import torch.nn as nn
import math
device='cuda'
class SINRDataset(Dataset):
    def __init__(self):
        data_path = "../gcp-server-dh2/ccp-vis/data/"
        self.img_paths = []
        self.metadata_paths = []
        self.n=int(8*0.8)
        for _ in tqdm(range(int(1e5))):
            for k in range(2,100):
                filename = f"{_}{k}"
                self.img_paths.append(data_path+f"{filename}_image.pt")
                self.metadata_paths.append(data_path+f"{filename}_metadata_train.pt")
    def __len__(self):
        return len(self.img_paths)*self.n
    def __getitem__(self, idx):
        import torch
        img = torch.load(self.img_paths[math.floor(idx/self.n)])
        meta = torch.load(self.metadata_paths[math.floor(idx/self.n)])[int(idx%self.n)]
        return dict(img=img,
                    loc=meta[:2]),meta[-1]
    
def create_dataloader(batch_size=128):
    return DataLoader(SINRDataset(),batch_size=batch_size, shuffle=True)

model = Net().to(device)
model = nn.DataParallel(model)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.MSELoss()

score = []
from tqdm import tqdm
for e in range(int(1e6)):
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
torch.save(model.state_dict(),f"model.pt")
import numpy as np
np.save("score.npy",score)