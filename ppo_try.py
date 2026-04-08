print('Hello World!')
import torch
import torch._inductor.config as inductor_config
inductor_config.cpp_wrapper = False 
torch.set_float32_matmul_precision('high')
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch.distributions as distributions
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"当前运行设备: {device}")
k=1.0

# def getDerivative(y,f):
#     costh=torch.cos(y[...,1])
#     sinth=torch.sin(y[...,1])
#     D = 4/3*(1+k)-costh**2
#     xdot=(4/3*y[...,2]-costh*y[...,3])/D
#     thdot=((1+k)*y[...,3]-costh*y[...,2])/D
#     pxdot=f
#     pthdot=sinth*(1-xdot*thdot)
#     return torch.stack([xdot,thdot,pxdot,pthdot],dim=-1)
# def rk2solver(y0,dt,f):
#     ymid=y0+dt/2*getDerivative(y0,f)
#     return y0+dt*getDerivative(ymid,f)
# @torch.compile
def rk2solver(y0,dt,f):
    y=y0
    costh=torch.cos(y[...,1])
    sinth=torch.sin(y[...,1])
    D = 4/3*(1+k)-costh**2
    xdot=(4/3*y[...,2]-costh*y[...,3])/D
    thdot=((1+k)*y[...,3]-costh*y[...,2])/D
    pxdot=f
    pthdot=sinth*(1-xdot*thdot)
    v1=torch.stack([xdot,thdot,pxdot,pthdot],dim=-1)
    ymid=y0+dt/2*v1
    # return y0+dt*getDerivative(ymid,f)

    y=ymid
    costh=torch.cos(y[...,1])
    sinth=torch.sin(y[...,1])
    D = 4/3*(1+k)-costh**2
    xdot=(4/3*y[...,2]-costh*y[...,3])/D
    thdot=((1+k)*y[...,3]-costh*y[...,2])/D
    pxdot=f
    pthdot=sinth*(1-xdot*thdot)
    v2=torch.stack([xdot,thdot,pxdot,pthdot],dim=-1)
    y=y0+dt*v2
    return y

def reward(y,f):
    # f=f.view_as(y[...,0])
    return 5.0-(y[...,0]**2*0.5+torch.relu(y[...,0]**2-16)*1.5+(1-torch.cos(y[...,1]))*5+0.05*y[...,2]**2+1*y[...,3]**2+0.005*f**2)
class Actor(nn.Module):
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
        # self.critic=nn.Sequential(
        #     nn.Linear(256,1)
        # )
        self.log_std=nn.Parameter(torch.zeros(1))
    def forward(self,state)-> tuple[distributions.Normal, torch.Tensor]:
        feat=self.rootnet(state)
        mu=10*self.actor(feat)
        # v=self.critic(feat)
        # std=torch.exp(self.log_std)
        # dist=distributions.Normal(mu,std)
        return mu,self.log_std
class Critic(nn.Module):
    def __init__(self,indim):
        super().__init__()
        self.rootnet=nn.Sequential(
                nn.Linear(indim,256),
                nn.LeakyReLU(),
                nn.Linear(256,256),
                nn.LeakyReLU()
        )
        self.critic=nn.Sequential(
            nn.Linear(256,1),
        )
        # self.critic=nn.Sequential(
        #     nn.Linear(256,1)
        # )
        self.log_std=nn.Parameter(torch.zeros(1))
    def forward(self,state)-> tuple[distributions.Normal, torch.Tensor]:
        feat=self.rootnet(state)
        v=self.critic(feat)
        # std=torch.exp(self.log_std)
        # dist=distributions.Normal(mu,std)
        return v.squeeze(-1)

def chkdeath(y):
    x=torch.abs(y[...,0])>5.0
    y=torch.abs(y[...,1])>6.28
    z=x or y
    return z.float()

mainNetwork=Actor(7).to(device)
# mainNetwork.load_state_dict(torch.load("pendulum_controller_ppo.pth"))
mainNetwork=torch.compile(mainNetwork)
mainNetwork2=Critic(7).to(device)
# mainNetwork2.load_state_dict(torch.load("pendulum_controller_critic.pth"))
mainNetwork2=torch.compile(mainNetwork2,mode="reduce-overhead")
optimizer=optim.Adam(mainNetwork.parameters(),lr=0.0003)
optimizer2=optim.Adam(mainNetwork2.parameters(),lr=0.001)

sample_batch_size=4096
N=2 
level=0
max_level=10
dt=0.05
steps=200

def get_init(batch_size):
    x_scale = 0.1 + (level / max_level) * 2.0
    x_pos = (torch.rand(batch_size, device=device) - 0.5) * 2.0 * x_scale
    th_scale = 0.1 + (level / max_level) * 3.04
    th_pos = (torch.rand(batch_size, device=device) - 0.5) * 2.0 * th_scale
    v_noise = (torch.rand((batch_size,2), device=device) - 0.5) * 0.2
    return torch.stack([x_pos, th_pos, v_noise[:,0], v_noise[:,1]],dim=-1)

# buffer=[]
gamma=0.95
lamb=0.95
K=7
MAX_EPISODE_NUMBER=50
mini_batch_size=1024
e=0.2
c1=0.5
c2=0.02
u=N*sample_batch_size*steps//mini_batch_size
# -------------主训练循环开始，采样----------------
# for n in range(N):
#     y0=get_init(sample_batch_size)
#     trajectory=[]
#     for i in range(steps):
#         state_in=torch.stack([y0[:,0],torch.sin(y0[:,1]),torch.cos(y0[:,1]),torch.sin(2*y0[:,1]),torch.cos(2*y0[:,1]),y0[:,2],y0[:,3]],dim=-1)
#         dist,V=mainNetwork(state_in)
#         f=dist.sample().squeeze(-1)   #f (256, )
#         log_prob=dist.log_prob(f)     #log_prob (256, )
#         next_state=rk2solver(y0,dt,f)
#         rwd=reward(next_state,f)     # rwd (256, )
#         mask_death=chkdeath(next_state)   #mask_death (256, )
#         # rwd=rwd-mask_death*100.0
#         trajectory.append({'state':state_in,'action':f,'reward':rwd,'log_prob':log_prob.detach(),'V':V.detach(),'death':mask_death})
#         if i==steps-1:
#             trajectory[-1]['next_state']=next_state
#         reset_states=get_init(sample_batch_size)
#         y0 = torch.where(mask_death.unsqueeze(-1)> 0.5, reset_states, next_state)
#     buffer.append(trajectory)

st=torch.zeros([sample_batch_size,steps*N,7],device=device)
ac=torch.zeros((sample_batch_size,steps*N),device=device)
rw=torch.zeros((sample_batch_size,steps*N),device=device)
lp=torch.zeros((sample_batch_size,steps*N),device=device)
vv=torch.zeros((sample_batch_size,steps*N),device=device)
dth=torch.zeros((sample_batch_size,steps*N),device=device)
ns=torch.zeros((sample_batch_size,steps*N,4),device=device)
A=torch.zeros((sample_batch_size,steps*N),device=device)
R=torch.zeros((sample_batch_size,steps*N),device=device)
buffer={'state':st,'action':ac,'reward':rw,'log_prob':lp,'V':vv,'death':dth,'next_state':ns,'A':A,'R':R}

def reshap(tensor):
    return tensor.reshape(-1) if tensor.dim()==2 else tensor.reshape(-1,tensor.shape[-1])

def shfl(tensor,shflidx):
    return tensor[shflidx,:] if tensor.dim()==2 else tensor[shflidx]



@torch.compile
def sample_step(y0, net_mu, net_lgstd):
    f=torch.clamp(net_mu+torch.randn_like(net_mu)*torch.exp(net_lgstd),-15,15)
    log_prob=-((f-net_mu)**2)/(2*torch.exp(2*net_lgstd))-net_lgstd-0.9189385332
    f_sq=f.squeeze(-1)
    next_state=rk2solver(y0,dt,f_sq)
    rwd=reward(next_state,f_sq)
    m1=torch.abs(next_state[...,0])>5.0
    m2=torch.abs(next_state[...,2])>2.5
    mask_death=torch.clip((m1+m2).float(),max=1)
    rwd = rwd - mask_death * 100.0
    return f_sq, log_prob.squeeze(-1), next_state, rwd, mask_death


@torch.compile(mode='reduce-overhead')
def compute_loss(state_b,returns_b,action_b,log_prob_b,A_b):    
    muu,lggstd=mainNetwork(state_b)
    muu=muu.squeeze(-1)
    V_pred=mainNetwork2(state_b)
    new_log_probs=-((action_b-muu)**2)/(2*torch.exp(2*lggstd))-lggstd-0.9189385332
    ratio=torch.clamp(torch.exp(new_log_probs-log_prob_b),0,5)
    surr1=ratio*A_b
    surr2=torch.clamp(ratio,1-e,1+e)*A_b
    actor_loss=-torch.mean(torch.min(surr1,surr2))
    critic_loss=torch.mean((V_pred-returns_b)**2)
    entropy_loss=-torch.mean(lggstd)-1.418938533
    return actor_loss + c1 * critic_loss + c2 * entropy_loss




# -------采样--------
for _ in range(MAX_EPISODE_NUMBER):
    print(f'-------Episode {_}-------')
    print('sampling start.')
    with torch.no_grad():
        for n in range(N):
            y0=get_init(sample_batch_size)
    # trajectory=[]
            for i in range(steps):
                # torch.compiler.cudagraph_mark_step_begin() 
                state_in=torch.stack([y0[:,0],torch.sin(y0[:,1]),torch.cos(y0[:,1]),torch.sin(2*y0[:,1]),torch.cos(2*y0[:,1]),y0[:,2],y0[:,3]],dim=-1)
                mu,lgstd=mainNetwork(state_in)
                # mu=mu.clone()
                V=mainNetwork2(state_in)
                # f=torch.clamp(mu+torch.randn_like(mu)*torch.exp(lgstd),-15,15)
                # log_prob=-((f-mu)**2)/(2*torch.exp(2*lgstd))-lgstd-0.9189385332
                # # f=dist.sample().squeeze(-1)   #f (256, )
                # # log_prob=dist.log_prob(f.unsqueeze(-1)).squeeze(-1)     #log_prob (256, )
                # next_state=rk2solver(y0,dt,f.squeeze(-1))
                # rwd=reward(next_state,f.squeeze(-1))     # rwd (256, )
                # mask_death=chkdeath(next_state)   #mask_death (256, )
            # rwd=rwd-mask_death*100.0
            # trajectory.append({'state':state_in,'action':f,'reward':rwd,'log_prob':log_prob.detach(),'V':V.detach(),'death':mask_death})
                fsq,log_prob,next_state,rwd,mask_death=sample_step(y0,mu,lgstd)
                buffer['state'][:,i+n*steps,:]=state_in
                buffer['action'][:,i+n*steps]=fsq
                buffer['reward'][:,i+n*steps]=rwd.squeeze(-1)
                buffer['log_prob'][:,i+n*steps]=log_prob.detach()
                buffer['V'][:,i+n*steps]=V.detach().squeeze(-1)
                buffer['death'][:,i+n*steps]=mask_death
                if i==steps-1:
                    buffer['next_state'][:,i+n*steps,:]=next_state
                reset_states=get_init(sample_batch_size)
                y0 = torch.where(mask_death.unsqueeze(-1)> 0.5, reset_states, next_state)
    # buffer.append(trajectory)
    print('sampling succeeded.')
# --------计算GAE------------
    with torch.no_grad():
        for n in range(N):
    # traj=buffer[n]
            final_state_raw=buffer['next_state'][:,(n+1)*steps-1,:]
            final_state_in=torch.stack([final_state_raw[:,0],torch.sin(final_state_raw[:,1]),
                                    torch.cos(final_state_raw[:,1]),torch.sin(2*final_state_raw[:,1]),
                                    torch.cos(2*final_state_raw[:,1]),final_state_raw[:,2],final_state_raw[:,3]],dim=-1)
            Vp=mainNetwork2(final_state_in)
            A=torch.zeros(sample_batch_size,device=device)
            for t in reversed(range(steps)):
        # V_next=traj[t+1]['V'] if t+1<steps else Vp
                V_next=buffer['V'][...,n*steps+t+1] if t+1<steps else Vp
                delta=buffer['reward'][...,n*steps+t]+gamma*(1-buffer['death'][...,n*steps+t])*V_next-buffer['V'][...,n*steps+t]
                A=delta+gamma*lamb*A*(1-buffer['death'][:,n*steps+t])
                buffer['A'][:,n*steps+t]=A
                buffer['R'][:,n*steps+t]=A+buffer['V'][:,n*steps+t]

    adv_mean=buffer['A'].mean()
    adv_std=buffer['A'].std()
    buffer['A']=(buffer['A']-adv_mean)/(adv_std+1e-8)

# -----------GAE计算结束------------
    print('compute gae succeed.')


    for epoch in range(K):
        
        buffer_shfled={}
        shfl_idx=torch.randperm(N*sample_batch_size*steps,device=device)
        for key,value in buffer.items():
            buffer_shfled[key]=reshap(value)
        ls=[]
        for i in range(u):
            start = i * mini_batch_size
            end = (i + 1) * mini_batch_size

            batch_idx=shfl_idx[start:end]
            b_state=buffer_shfled['state'][batch_idx,:]
            # muu,lggstd,V_pred=mainNetwork(b_state)
            b_returns=buffer_shfled['R'][batch_idx]
            b_actions=buffer_shfled['action'][batch_idx]
            b_old_log_prob=buffer_shfled['log_prob'][batch_idx]
            b_A=buffer_shfled['A'][batch_idx]
            total_loss=compute_loss(b_state,b_returns,b_actions,b_old_log_prob,b_A)
            # old_action=b_actions
            # # new_log_probs=dis.log_prob(old_action.unsqueeze(-1)).squeeze(-1)
            # new_log_probs=-((old_action-muu)**2)/(2*torch.exp(2*lggstd))-lggstd-0.9189385332
            # ratio=torch.clamp(torch.exp(new_log_probs-b_old_log_prob),0,5)
            # surr1=ratio*b_A
            # surr2=torch.clamp(ratio,1-e,1+e)*b_A
            # actor_loss=-torch.mean(torch.min(surr1,surr2))
            # critic_loss=torch.mean((V_pred-returns)**2)
            # entropy_loss=-torch.mean(lggstd)-1.418938533
            # total_loss = actor_loss + c1 * critic_loss + c2 * entropy_loss
            optimizer.zero_grad(set_to_none=True)
            optimizer2.zero_grad(set_to_none=True)
            # if torch.isnan(total_loss) or torch.isinf(total_loss):
            #     print('NaN detected,skipping batch.')
            #     optimizer.zero_grad(set_to_none=True)
            #     continue
            # print('doing backward.')
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(mainNetwork.parameters(), 0.5)  # 梯度裁剪
            optimizer.step()
            optimizer2.step()
            ls.append(total_loss.item())
            # print(total_loss.item())
        v=sum(ls)/u
        print(f'epoch {epoch} ,loss {v}')
    uu=buffer['reward'].mean().item()
    print(f'\nAverage Reward={uu}')
    if uu>4:
        level+=0.05
    print('level=',level)
torch.save(mainNetwork._orig_mod.state_dict(), "pendulum_controller_ppo.pth")
torch.save(mainNetwork2._orig_mod.state_dict(), "pendulum_controller_critic.pth")
print("权重已成功保存！")