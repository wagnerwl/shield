# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoundDetectorCNN(nn.Module):
    def __init__(self):
        super(SoundDetectorCNN, self).__init__()
        
        # Block 1
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16) # NEU: Normalisiert die 16 Feature-Maps
        self.pool1 = nn.MaxPool2d(2)
        
        # Block 2
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32) # NEU: Normalisiert die 32 Feature-Maps
        self.pool2 = nn.MaxPool2d(2)
        
        # Fully Connected
        self.fc1 = nn.Linear(5120, 64) 
        self.dropout = nn.Dropout(p=0.6) 
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        # Reihenfolge ist wichtig: Conv -> BatchNorm -> ReLU -> Pooling
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool2(x)
        
        x = torch.flatten(x, 1) 
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x) 
        x = torch.sigmoid(self.fc2(x)) 
        return x