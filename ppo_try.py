import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.distributions as distributions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前运行设备: {device}")
k=1.0

def getDerivative(y,f):
    costh=torch.cos(y[...,1])
    sinth=torch.sin(y[...,1])
    D = 4/3*(1+k)-costh**2
    xdot=(4/3*y[...,2]-costh*y[...,3])/D
    thdot=((1+k)*y[...,3]-costh*y[...,2])/D
    pxdot=f
    pthdot=sinth*(1-xdot*thdot)
    return torch.stack([xdot,thdot,pxdot,pthdot],dim=-1)
def rk2solver(y0,dt,f):
    ymid=y0+dt/2*getDerivative(y0,f)
    return y0+dt*getDerivative(ymid,f)
def reward(y,f):
    # f=f.view_as(y[...,0])
    return 1.0-(y[...,0]**2*0.5+torch.relu(y[...,0]**2-16)+(1-torch.cos(y[...,1]))*5+0.05*y[...,2]**2+0.6*y[...,3]**2+0.005*f**2)
class ActorCritic(nn.Module):
    def __init__(self,indim):
        super().__init__()
        self.rootnet=nn.Sequential(
                nn.Linear(indim,256),
                nn.LeakyReLU(),
                nn.Linear(256,256),
                nn.LeakyReLU()
        )
        self.actor=nn.Sequential(
            nn.Linear(256,1),
            nn.Tanh()
        )
        self.critic=nn.Sequential(
            nn.Linear(256,1)
        )
        self.log_std=nn.Parameter(torch.zeros(1))
    def forward(self,state)-> tuple[distributions.Normal, torch.Tensor, torch.Tensor]:
        feat=self.rootnet(state)
        mu=10*self.actor(feat)
        v=self.critic(feat)
        std=torch.exp(self.log_std)
        dist=distributions.Normal(mu,std)
        return dist,v

def chkdeath(y):
    x=torch.abs(y[...,0])>5.0
    return x.float()

mainNetwork=ActorCritic(7).to(device)


sample_batch_size=256
N=8
#先不加for循环
level=0
max_level=10
dt=0.05
steps=50
# x_scale = 0.1 + (level / max_level) * 2.0
# x_pos = (torch.rand(1, device=device) - 0.5) * 2.0 * x_scale
# th_scale = 0.1 + (level / max_level) * 3.04
# th_pos = (torch.rand(1, device=device) - 0.5) * 2.0 * th_scale
def get_init(batch_size):
    x_scale = 0.1 + (level / max_level) * 2.0
    x_pos = (torch.rand(batch_size, device=device) - 0.5) * 2.0 * x_scale
    th_scale = 0.1 + (level / max_level) * 3.04
    th_pos = (torch.rand(batch_size, device=device) - 0.5) * 2.0 * th_scale
    v_noise = (torch.rand((batch_size,2), device=device) - 0.5) * 0.2
    return torch.stack([x_pos, th_pos, v_noise[:,0], v_noise[:,1]],dim=-1)

buffer=[]
gamma=0.95
lamb=0.95
# A=torch.zeros((sample_batch_size,1),device=device)
# -------------主训练循环开始，采样----------------
# for n in range(N):
#     y0=get_init(sample_batch_size)
#     trajectory=[]
#     for i in range(steps):
#         state_in=torch.stack([y0[:,0],torch.sin(y0[:,1]),torch.cos(y0[:,1]),torch.sin(2*y0[:,1]),torch.cos(2*y0[:,1]),y0[:,2],y0[:,3]],dim=-1)
#         dist,V=mainNetwork(state_in)
#         f_raw=dist.sample()
#         log_prob=dist.log_prob(f_raw)
#         f=f_raw.squeeze(-1)
#         log_prob=log_prob.squeeze(-1)
#         V=V.squeeze(-1)
#         next_state=rk2solver(y0,dt,f)
#         rwd=reward(next_state,f)
#         mask_death=chkdeath(next_state)
#         trajectory.append({'state':state_in,'action':f_raw,'reward':rwd,'log_prob':log_prob.detach(),'V':V.detach(),'death':mask_death})
#         reset_states=get_init(sample_batch_size)
#         y0 = torch.where(mask_death.unsqueeze(-1) > 0.5, reset_states, next_state)
# # ------------一轮采样结束，立即计算优势------------
#     final_state_in=torch.stack([y0[:,0],torch.sin(y0[:,1]),torch.cos(y0[:,1]),torch.sin(2*y0[:,1]),torch.cos(2*y0[:,1]),y0[:,2],y0[:,3]],dim=-1)
#     with torch.no_grad:
#         _,Vp=mainNetwork(trajectory[steps-1]['state'])
#         Vp=Vp.squeeze(-1)
#     A=torch.zeros(sample_batch_size,device=device)
#     for t in reversed(range(steps)):
#         # _,Vp=mainNetwork(trajectory[steps-1]['state'])
#         V_plus_one=buffer[t+1]['V'] if t+1 < steps else Vp
#         delta=trajectory[t]['reward']+gamma*(1-trajectory[t]['death'])*V_plus_one-trajectory[t]['V']
#         A=delta+gamma*lamb*A*(1-trajectory[t]['death'])
#         trajectory[t]['A']=A
#         trajectory[t]['R']=A+trajectory[t]['V']
#     buffer.append(trajectory)