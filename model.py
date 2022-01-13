import os
import json
import base64
from io import BytesIO

import torch
from torch.utils.data import Dataset
from torch import nn
from PIL import Image

class TuxDriverDataset(Dataset):
    def __init__(self, folder_path, transform=None, target_transform=None):
        self.dataset_path = folder_path
        self.data = []
        for filename in os.listdir(self.dataset_path):
            if ".json" in filename:
                with open(self.dataset_path + "/" + filename) as jsonfile:
                    self.data += json.loads(jsonfile.read())

        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_stream = BytesIO(base64.b64decode(self.data[idx]["frame"]))
        image = Image.open(image_stream).resize((64, 36))

        if self.transform is not None:
            image = self.transform(image)
        
        return image, torch.tensor(self.data[idx]["expected"]).float()

class TuxDriverModel(nn.Module):
    def __init__(self):
        super(TuxDriverModel, self).__init__()

        self.convolution = nn.Sequential(nn.Conv2d(1, 1, (8,8)), nn.ReLU(), nn.MaxPool2d((2,2)), nn.Conv2d(1, 1, (4,4)), nn.ReLU())

        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(nn.Linear(275, 100), nn.ReLU(),
                                    nn.Linear(100, 100), nn.ReLU(),
                                    nn.Linear(100, 100), nn.ReLU(),
                                    nn.Linear(100, 100), nn.ReLU(),
                                    nn.Linear(100, 100), nn.ReLU(),
                                    nn.Dropout(),
                                    nn.Linear(100, 4), nn.Sigmoid())

    def forward(self, x):
        out = self.convolution(x)
        out = self.flatten(out)
        out = self.linear(out)

        return out
