# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoundDetectorCNN(nn.Module):
    def __init__(self):
        super(SoundDetectorCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        
        # HIER ist die entscheidende Änderung für 1,28 Sekunden!
        # 32 (Kanäle) * 16 (Höhe) * 10 (Breite) = 5120
        self.fc1 = nn.Linear(in_features=5120, out_features=64) 
        self.fc2 = nn.Linear(in_features=64, out_features=1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        
        x = torch.flatten(x, 1) 
        
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x)) 
        
        return x