import matplotlib.pyplot as plt
import numpy as np
from functions import heun_kpz
import pandas as pd
from time import time

lambd=1
v=1
# dt=0.1
dt=[0.025,0.05,0.075,0.1]
dx=np.sqrt(2*v*dt)
tf, nsteps=5000, 10
# N=(1/dx)*np.array([50,100,150,200])

D=1/2
nrep=100
# wn=np.zeros((len(N),int(tf/(dt*nsteps)+1)))
wn=[[],[],[],[]]
err=wn.copy()
# nsaved=int(tf/(dt*nsteps))
t1=time()
for i,deltat in enumerate(dt):
    N=(1/dx[i])*150
    wall=np.zeros((nrep,int(tf/(deltat*nsteps)+1)))

    for rep in range(nrep):
        print(f'dt={deltat}, experiment number: {rep+1}')
        nsaved,L=heun_kpz(f"files/prueba_{deltat}_{rep+1}.txt", lambd, v, D, deltat, dx[i],  tf, nsteps, N)
        data=pd.read_csv(f"files/prueba_{deltat}_{rep+1}.txt", header=None)
        w=np.zeros(nsaved+1)
        for j in range(nsaved+1):       
            h=np.fromstring(data[1][j], sep=' ')
            hmean=h.mean()
            h2mean=(h**2).mean()
            w[j]=h2mean-hmean**2
        wall[rep]=w
    wn[i]=wall.mean(axis=0)
    err[i]=wall.std(axis=0)
print(f'Total simulation time: {time()-t1} seconds')

#%% Variable lattice size plotting
time=np.array(data[0])
cond=time>1
markers=["s", "v", "D", "*"]
# colors=["black", "blue", "red", "yellow"]
errcolor=["cornflowerblue", "tan", "palegreen", "tomato"]
fig, ax=plt.subplots(figsize=(8,6))
ax.set(title=f"Width of the surface for $\\lambda=${lambd}", xlabel='$t$', ylabel='$w^2(L,t)$')
for i,n in enumerate(N):
    ax.errorbar(time[cond], wn[i][cond], yerr=err[i][cond]/np.sqrt(nrep), ecolor=errcolor[i], errorevery=3, fmt=markers[i], ms=1, label=f'L={round(n*dx)}')

ax.plot(time[cond][0:15],(1/6)*time[cond][0:15]**(2/3), ls='--', color='black', label='$\propto t^{2\\beta}$')
xm,xM=ax.get_xlim()
for i,n in enumerate(N):
    if n!=N[-1]:
        ax.axhline((1/45)*n*dx, xmin=(time[cond][3500]-xm)/(xM-xm),ls='dashdot', c='black')
    else:
        ax.axhline((1/45)*n*dx, xmin=(time[cond][3500]-xm)/(xM-xm),ls='dashdot', c='black', label='$\\propto L^{2\\alpha}$')
ax.legend()
ax.set_yscale('log')
ax.set_xscale('log')
fig.savefig(f"figures/wlambda_{lambd}.pdf", dpi=200, bbox_inches='tight')

#%% Variable time step plotting
markers=["s", "v", "D", "*"]
errcolor=["cornflowerblue", "tan", "palegreen", "tomato"]
fig, ax=plt.subplots(figsize=(8,6))
ax.set(title="Width of the surface, variable time step", xlabel='$t$', ylabel='$w^2(L,t)$')
for i, deltat in enumerate(dt):
    time=np.arange(0,tf+nsteps*deltat, nsteps*deltat)
    if i==2:
        time=np.delete(time,-1)
    cond=time>1
    ax.errorbar(time[cond], wn[i][cond], yerr=err[i][cond]/np.sqrt(nrep),ecolor=errcolor[i],fmt=markers[i], ms=1,label=f'$\\Delta t$={deltat}')
ax.legend()
ax.set_yscale('log')
ax.set_xscale('log')
fig.savefig("figures/variable_dt.pdf", dpi=200, bbox_inches='tight')
#%%
betas=np.zeros(N.shape[0])
errorb=betas.copy()
alphas=betas.copy()
erroras=betas.copy()

tsat=3500
nt=int(time[-1])-tsat
wl=betas.copy()

for i,n in enumerate(N): #alpha calculation
    wl[i]=(1/nt)*sum(wn[i][tsat::])
popt, pcov=np.polyfit(np.log(N*dx), np.log(wl),1, cov=True)
alpha=popt[0]
errora=np.sqrt(np.diag(pcov))[0]   

for i,n in enumerate(N): #betas calculation
    popt, pcov=np.polyfit(np.log(time[cond])[2:5], np.log(wn[i][cond])[2:5],1, cov=True)
    betas[i]=popt[0]
    errorb[i]=np.sqrt(np.diag(pcov))[0]
    

    



   
