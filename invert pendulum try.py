import torch
import torch.nn as nn
import torch.optim as optim
class VerySimpleCar(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(5,256),
            nn.Tanh(),
            nn.Linear(256,256),
            nn.Tanh(),
            nn.Linear(256,1)
            nn.Tanh()
        )
    def getForce(self,y):
        return 10.0*self.net(y)
    