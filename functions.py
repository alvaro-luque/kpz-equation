import numpy as np
import time

def heun_kpz(file,lambd,v,D,dt,dx,tf,nsteps, N):
    
    
    L=N*dx
    Deff=np.sqrt(D*dt/dx)
    diff=v*dt/(dx**2)
    g=(1/8)*lambd*dt/(dx**2)
    nsaved=int(tf/(dt*nsteps))
    t=0
    
        
    rng=np.random.default_rng()
    h=np.zeros(int(N))
    
    with open(file,'w+') as f:
        t1=time.time()
        f.write(f'{t},{" ".join(map(str,h))}\n')
        for i in range(nsaved):
            for j in range(nsteps):
                hm, hp=np.roll(h,1), np.roll(h,-1)
                uv=Deff*rng.normal(size=len(h))
                aux=diff*(hp-2*h+hm)+g*(hp-hm)**2
                h1=h+aux+uv
                hm, hp=np.roll(h1,1), np.roll(h1,-1)
                h+=0.5*(aux+diff*(hp-2*h1+hm)+g*(hp-hm)**2)+uv
                t+=dt
            f.write(f'{t},{" ".join(map(str,h))}\n')
    
    print(f"Elapsed time: {time.time()-t1} seconds")
    return nsaved,L    
