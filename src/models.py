import numpy as np

import torch
import torch.nn as nn


class FCSM(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(in_features, 24),
            nn.ReLU(),
            nn.Linear(24, 48),
            nn.Softmax(dim=0)
        )
        # self.fc1 = nn.Linear(in_features, 24)
        # self.fc2 = nn.Linear(24, 48)
        # self.sm = nn.Softmax(dim=0)
    
    def forward(self, x: torch.Tensor):
        return self.model(x)
        # self.sm(self.fc(x))
    

if __name__ == '__main__':
    model = FCSM()
    pred = model(torch.tensor([1.0, 3.0])).detach().numpy()
    print(np.sum(pred))

