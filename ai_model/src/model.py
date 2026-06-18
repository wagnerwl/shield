# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SoundDetectorCNN(nn.Module):
    def __init__(self):
        super(SoundDetectorCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2)
        
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2)
        
        self.fc1 = nn.Linear(5120, 64) 
        
        # NEU: Dropout-Schicht! Deaktiviert zufällig 50% der Verbindungen
        self.dropout = nn.Dropout(p=0.6) 
        
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) 
        
        x = F.relu(self.fc1(x))
        
        # NEU: Dropout anwenden, bevor es zur finalen Entscheidung geht
        x = self.dropout(x) 
        
        x = torch.sigmoid(self.fc2(x)) 
        return x