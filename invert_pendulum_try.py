import torch
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.optim as optim
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前运行设备: {device}")
class VerySimpleCar(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(7,256),
            nn.LeakyReLU(),
            nn.Linear(256,256),
            nn.LeakyReLU(),
            nn.Linear(256,1),
            nn.Tanh()
        )
    def getForce(self,y):
        yy=torch.stack([y[0],torch.sin(y[1]),torch.cos(y[1]),torch.sin(2*y[1]),torch.cos(2*y[1]),y[2],y[3]])
        return 10.0*self.net(yy)
mu=1.0
def getDerivative(y,f):
    costh=torch.cos(y[1])
    sinth=torch.sin(y[1])
    D = 4/3*(1+mu)-costh**2
    xdot=(4/3*y[2]-costh*y[3])/D
    thdot=((1+mu)*y[3]-costh*y[2])/D
    pxdot=f
    pthdot=sinth*(1-xdot*thdot)
    return torch.stack([xdot,thdot,pxdot,pthdot])
def rk2solver(y0,dt,f):
    ymid=y0+dt/2*getDerivative(y0,f)
    return y0+dt*getDerivative(ymid,f)
def loss(y):
    return 0.01*y[0]**2+(1-torch.cos(y[1]))*2+0.001*y[2]**2+0.001*y[3]**2
net = VerySimpleCar().to(device)

optimizer = optim.Adam(net.parameters(), lr=0.001)
dt = 0.05  # 时间步长

steps=100
times=5
y0 = torch.tensor([0.8, 0.2, 0.0, 0.0], dtype=torch.float32, requires_grad=False,device=device)

def sim_step(y0,dt,step):
    ttlos = torch.tensor(0.0, device=y0.device)
    for i in range(step):
        F=net.getForce(y0)
        y=rk2solver(y0,dt,F[0])
        ttlos+=loss(y)
        y0=y
    return y,ttlos/step
compiled_step = torch.compile(sim_step, mode="reduce-overhead")
for epoch in range(250):
    optimizer.zero_grad()
    # y0 = torch.tensor([0.8, 0.2, 0.0, 0.0], dtype=torch.float32, requires_grad=False,device=device)
    ttloss=0
    # for i in range(steps):
    #     F=net.getForce(y0)
    #     y=rk2solver(y0,dt,F[0])
    #     ttloss+=loss(y)
    #     y0=y
    # ttloss/=steps
    y,ttloss=compiled_step(y0,dt,steps)
    # y,ttloss=sim_step(y0,dt,steps)
    ttloss.backward()
    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
    optimizer.step()
    # y0=y
    y0=y.clone().detach()
    if ttloss>10 or abs(y[1])>10 or abs(y[0])>10:
        y0 = torch.tensor([0.8, 0.2, 0.0, 0.0], dtype=torch.float32, requires_grad=False,device=device)
        print('---Reset---')
    print(ttloss)
    print(y)
    # print(F)
    print(epoch)