import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.distributions as distributions

k=1.0

def getDerivative(y,f):
    costh=torch.cos(y[1])
    sinth=torch.sin(y[1])
    D = 4/3*(1+k)-costh**2
    xdot=(4/3*y[2]-costh*y[3])/D
    thdot=((1+k)*y[3]-costh*y[2])/D
    pxdot=f
    pthdot=sinth*(1-xdot*thdot)
    return torch.stack([xdot,thdot,pxdot,pthdot])
def rk2solver(y0,dt,f):
    ymid=y0+dt/2*getDerivative(y0,f)
    return y0+dt*getDerivative(ymid,f)

class ActorCritic:
    def __init__(self,indim):
        self.rootnet=nn.Sequential(
                nn.Linear(7,256),
                nn.LeakyReLU(),
                nn.Linear(256,256),
                nn.LeakyReLU()
        )
        self.actor=nn.Sequential(
            nn.Linear(256,1),
            10*nn.Tanh()
        )
        self.critic=nn.Sequential(
            nn.Linear(256,1)
        )
        self.log_std=nn.Parameter(torch.zeros(1))
    def forward(self,state):
        feat=self.rootnet(state)
        mu=self.actor(feat)
        v=self.critic(feat)
        std=torch.exp(self.log_std)
        dist=distributions.Normal(mu,std)
        return dist,v
