"""het-K x unbiased spatial pairs: the 2x2 the diagnosis predicts.

Trajectory pooling carries an O(1/n) within-round bias (s_{i,k} shares noise
with the round-centred gbar). Het's groups are 8x smaller than homo's single
group, so het suffers ~8x the bias -- which is why it lost despite having
1.37-1.47x MORE design spread in the flat directions.

muse's construction removes that bias: centre across workers at a FIXED local
step, so the gradient noise is fresh relative to the position. Verified
separately at 8-10x bias reduction with no score cost.

Prediction: het x spatial wins the flat buckets, because the extra design
spread survives once the bias is gone.
"""
import math, random
D, N_OUT, ETA, SIG, REPS = 120, 10, 0.05, 1.0, 12
KHOMO, N = 8, 32
HET = [(16,2),(8,4),(4,8),(2,16),(2,32)]

def run_group(rng, lam, x, N_, K_, txx, txg, sxx, sxg):
    dev=[[0.0]*N_ for _ in range(D)]; pts=[[] for _ in range(D)]; grd=[[] for _ in range(D)]
    for _k in range(K_):
        sk=[[0.0]*N_ for _ in range(D)]; gk=[[0.0]*N_ for _ in range(D)]
        for j in range(D):
            lj,xj,dj=lam[j],x[j],dev[j]
            for i in range(N_):
                s=dj[i]; g=lj*(xj+s)+rng.gauss(0,SIG)
                pts[j].append(s); grd[j].append(g); sk[j][i]=s; gk[j][i]=g
                dj[i]=s-ETA*g
        if N_>1:                      # spatial: centre across workers at fixed k
            for j in range(D):
                ms=sum(sk[j])/N_; mg=sum(gk[j])/N_
                sxx[j]+=sum((v-ms)**2 for v in sk[j])
                sxg[j]+=sum((v-ms)*(w-mg) for v,w in zip(sk[j],gk[j]))
    n=N_*K_                            # trajectory: centre over all (i,k)
    for j in range(D):
        ms=sum(pts[j])/n; mg=sum(grd[j])/n
        txx[j]+=sum((p-ms)**2 for p in pts[j])
        txg[j]+=sum((p-ms)*(q-mg) for p,q in zip(pts[j],grd[j]))

def ols(xx,xg): return [xg[j]/xx[j] if xx[j]>1e-300 else 0.0 for j in range(D)]

rb=random.Random(0)
lam=[rb.uniform(8,12) for _ in range(N_OUT)]+[rb.uniform(0.1,1.0) for _ in range(D-N_OUT)]
flat=sorted(range(N_OUT,D),key=lambda j:lam[j]); h=len(flat)//2
buckets=[("sharp",list(range(N_OUT))),("flat high",flat[h:]),("flat LOW",flat[:h])]

res={}
for tag in ("homo","het"):
    ET=[0.0]*D; ES=[0.0]*D; n=0
    for r in range(REPS):
        rng=random.Random(500+r); x=[rng.gauss(0,1.0) for _ in range(D)]
        txx=[0.0]*D; txg=[0.0]*D; sxx=[0.0]*D; sxg=[0.0]*D
        if tag=="homo": run_group(rng,lam,x,N,KHOMO,txx,txg,sxx,sxg)
        else:
            for cnt,kk in HET: run_group(rng,lam,x,cnt,kk,txx,txg,sxx,sxg)
        ot=ols(txx,txg); os_=ols(sxx,sxg); n+=1
        for j in range(D):
            ET[j]+=abs(ot[j]-lam[j])/lam[j]; ES[j]+=abs(os_[j]-lam[j])/lam[j]
    res[tag]=([e/n for e in ET],[e/n for e in ES])

print("relative curvature error |lamhat-lam|/lam, one round from identical x")
print(f"  {'bucket':<11} | {'traj homo':>9} {'traj het':>9} | {'SPAT homo':>9} {'SPAT het':>9} | {'best':>12}")
for nm,idx in buckets:
    th=sum(res['homo'][0][j] for j in idx)/len(idx); tt=sum(res['het'][0][j] for j in idx)/len(idx)
    sh=sum(res['homo'][1][j] for j in idx)/len(idx); st=sum(res['het'][1][j] for j in idx)/len(idx)
    vals={'traj homo':th,'traj het':tt,'spat homo':sh,'spat het':st}
    best=min(vals,key=vals.get)
    print(f"  {nm:<11} | {th:>9.3f} {tt:>9.3f} | {sh:>9.3f} {st:>9.3f} | {best:>12}")
print(f"\nworker-steps: homo {N*KHOMO}, het {sum(c*k for c,k in HET)}")
