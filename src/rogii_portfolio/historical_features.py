"""Frozen historical 121-column FAST-SAFE feature builder.

GENERATED FILE.  The numerical body is the target-isolated transformation of
the single audited feature-builder cell in ``roggii_baseline.ipynb``.  It is
kept separate from :mod:`rogii_portfolio.features`, whose readable clean-room
builder has the same schema but deliberately simpler formulas.

The historical body reads only ``MD, X, Y, Z, GR, TVT_input`` from the
horizontal file and ``TVT, GR`` from the type well.  It never reads suffix
truth.  Numba's RNG is reset from a stable SHA-256 well seed before each pair
of particle filters; the historical particle/beam constants are unchanged.
"""

from __future__ import annotations

import hashlib
import multiprocessing
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from numba import njit

from .features import SAFE_FEATURES as HISTORICAL_FEATURE_ORDER


HISTORICAL_ORIGINAL_BUILDER_SHA256 = "500c9255b3bfd2268c8bb2e9be1a4025f0aa01d3483e67b4e28fa31e1d5d2c2c"
HISTORICAL_SAFE_SOURCE_SHA256 = "0e5f184ceee12abd4a1c4ce560dd544c009a6a5689a8430be7c476b15fcaf6f3"
HISTORICAL_BASE_SEED = 42
_FEATURE_BASE_SEED = HISTORICAL_BASE_SEED
_SAFE_HORIZONTAL_OBSERVABLES = ["MD", "X", "Y", "Z", "GR", "TVT_input"]
_SAFE_TYPEWELL_OBSERVABLES = ["TVT", "GR"]
# The frozen notebook body used set arithmetic for its allow-list.  Preserve
# that internal API while retaining one canonical sorted order for matrices.
SAFE_FEATURES = set(HISTORICAL_FEATURE_ORDER)


def _stable_well_seed(base_seed: int, well_id: str) -> int:
    if not isinstance(base_seed, int) or not 0 <= base_seed <= 0xFFFFFFFF:
        raise ValueError("base_seed must be a uint32")
    well = str(well_id)
    if not well or "\x00" in well:
        raise ValueError("well_id must be non-empty and cannot contain NUL")
    payload = f"{int(base_seed)}:{well}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "little")


@njit(cache=False)
def _seed_fast_safe_feature_rng(seed: int) -> None:
    np.random.seed(seed)


# Compile the explicit seed reset before any builder call or worker scheduling.
_seed_fast_safe_feature_rng(0)

# <BEGIN FROZEN HISTORICAL SAFE SOURCE>
SEED=42
NCPU=1

FORMATIONS=["ANCC","ASTNU","ASTNL","EGFDU","EGFDL","BUDA"]
PLANE_K=10; DENSE_SPW=60; DENSE_K=20; N_SPLITS=5

BEAMS=[
    (10,20.0,144.0,2,"cons"),
    (10, 8.0, 64.0,2,"loose"),
    ( 8,35.0,220.0,1,"vcons"),
    (10,14.0, 90.0,5,"sm5"),
    (20, 4.0, 36.0,3,"vloose"),
    (12,12.0,100.0,3,"mid"),
    (15,25.0,180.0,2,"stiff"),
]

PF_N=600; ANCC_N=600
PF_MOM=0.993; PF_VN=0.005; PF_PN=0.01
PF_GR_SIG_MIN=10.; PF_GR_SIG_MAX=60.; PF_GR_SIG_DEF=30.
PF_INIT_V_STD=0.02; PF_INIT_SPR=0.5; PF_RESAMP=0.5
PF_ROUGH_P=0.2; PF_ROUGH_V=0.003; PF_GR_WIN=5; PF_GR_WT=0.3
ANCC_ALPHA=0.998; ANCC_RN=0.002; ANCC_PN=0.005
ANCC_IR=0.01; ANCC_IS=0.3; ANCC_RP=0.1; ANCC_RR=0.001

@njit(cache=False)
def _interp1(grid, v, vmin, step):
    i = int((v - vmin) / step)
    if i < 0: return grid[0]
    n = len(grid) - 1
    if i >= n: return grid[n]
    t = (v - vmin) / step - i
    return grid[i]*(1.-t) + grid[i+1]*t

@njit(cache=False)
def _resamp(pos, aux, w, N, rp, rv):
    cum = np.zeros(N+1)
    for j in range(N): cum[j+1]=cum[j]+w[j]
    u0=np.random.uniform(0.,1./N)
    np2=np.empty(N); na=np.empty(N); ci=0
    for j in range(N):
        u=u0+j/N
        while ci<N-1 and cum[ci+1]<u: ci+=1
        np2[j]=pos[ci]+rp*np.random.randn()
        na[j] =aux[ci]+rv*np.random.randn()
    return np2,na

@njit(cache=False)
def _beam_jit(sgr, tw_gr, si, BS, mc, es):
    """Beam search Ã‚Â±2 delta, Numba JIT."""
    n=len(sgr); nt=len(tw_gr); MAX=BS*6
    bidx=np.zeros(BS,np.int64); bidx[0]=si
    bcost=np.full(BS,1e30);     bcost[0]=0.; bn=np.int64(1)
    hI=np.zeros((n,BS),np.int64); hP=np.zeros((n,BS),np.int64)
    cI=np.zeros(MAX,np.int64); cC=np.full(MAX,1e30); cP=np.zeros(MAX,np.int64)
    for step in range(n):
        gv=sgr[step]; nc=np.int64(0)
        for bi in range(bn):
            idx=bidx[bi]; cost=bcost[bi]
            for d in range(-2,3):            # Ã‚Â±2: TVT can go down
                ni=idx+d
                if ni<0 or ni>=nt: continue
                tot=cost+(gv-tw_gr[ni])**2/es+mc*(d if d>=0 else -d)
                fnd=np.int64(-1)
                for ci in range(nc):
                    if cI[ci]==ni: fnd=ci; break
                if fnd>=0:
                    if tot<cC[fnd]: cC[fnd]=tot; cP[fnd]=bi
                else:
                    if nc<MAX: cI[nc]=ni; cC[nc]=tot; cP[nc]=bi; nc+=1
        kept=min(BS,nc)
        for i in range(kept):
            mi=i
            for j in range(i+1,nc):
                if cC[j]<cC[mi]: mi=j
            if mi!=i:
                cI[i],cI[mi]=cI[mi],cI[i]
                cC[i],cC[mi]=cC[mi],cC[i]
                cP[i],cP[mi]=cP[mi],cP[i]
        hI[step,:kept]=cI[:kept]; hP[step,:kept]=cP[:kept]
        bidx[:kept]=cI[:kept]; bcost[:kept]=cC[:kept]; bn=kept
    best=np.int64(0)
    for b in range(1,bn):
        if bcost[b]<bcost[best]: best=b
    path=np.zeros(n,np.int64); b=best
    for s in range(n-1,-1,-1): path[s]=hI[s,b]; b=hP[s,b]
    return path

@njit(cache=False)
def _pf_ancc(md_v,z_v,gr_v,gg,vmin,step,gs,ls,ir,N,
              ALPHA,RN,PN,IS,RP,RR,RESAMP):
    pos=np.empty(N); rate=np.empty(N); w=np.ones(N)/N
    for j in range(N):
        pos[j]=ls+IS*np.random.randn()
        rate[j]=ir+0.01*np.random.randn()
    pts=np.empty(len(md_v)); std_=np.empty(len(md_v)); pm=md_v[0]-1.
    for i in range(len(md_v)):
        dm=md_v[i]-pm; dm=max(dm,1.)
        for j in range(N):
            rate[j]=ALPHA*rate[j]+RN*np.random.randn()
            pos[j]+=rate[j]*dm+PN*np.random.randn()
            tvt_j=pos[j]-z_v[i]
            tvt_j=max(tvt_j,vmin-50.); tvt_j=min(tvt_j,vmin+len(gg)*step+50.)
            pos[j]=tvt_j+z_v[i]
        if not np.isnan(gr_v[i]):
            ws=0.
            for j in range(N):
                eg=_interp1(gg,pos[j]-z_v[i],vmin,step)
                d=(gr_v[i]-eg)/gs
                lk=max(np.exp(-0.5*d*d) if d*d<600. else 0.,1e-300)
                w[j]*=lk; ws+=w[j]
            if ws>0.:
                for j in range(N): w[j]/=ws
            else:
                for j in range(N): w[j]=1./N
        ne=0.
        for j in range(N): ne+=w[j]*w[j]
        if 1./ne<RESAMP*N:
            pos,rate=_resamp(pos,rate,w,N,RP,RR)
            for j in range(N): w[j]=1./N
        tv=0.
        for j in range(N): tv+=w[j]*(pos[j]-z_v[i])
        pts[i]=tv; va=0.
        for j in range(N): va+=w[j]*(pos[j]-z_v[i]-tv)**2
        std_[i]=va**0.5; pm=md_v[i]
    return pts,std_

@njit(cache=False)
def _pf_z(md_v,z_v,gr_v,gr_sm_v,gg_p,gg_s,vmin,step,
          gs,ip,iv,beta,icpt,zsig,N,
          MOM,VN,PN,GR_WT,RP,RV,RESAMP):
    pos=np.empty(N); vel=np.empty(N); w=np.ones(N)/N
    for j in range(N):
        pos[j]=ip+0.5*np.random.randn()
        vel[j]=iv+0.02*np.random.randn()
    pts=np.empty(len(md_v)); std_=np.empty(len(md_v)); pm=md_v[0]-1.; pz=z_v[0]-1.
    for i in range(len(md_v)):
        dm=md_v[i]-pm; dm=max(dm,1.)
        dzd=(z_v[i]-pz)/dm; ve=beta*dzd+icpt
        for j in range(N):
            vel[j]=MOM*vel[j]+VN*np.random.randn()
            pos[j]+=vel[j]*dm+PN*np.random.randn()
            pos[j]=max(pos[j],vmin-50.); pos[j]=min(pos[j],vmin+len(gg_p)*step+50.)
        if not np.isnan(gr_v[i]):
            ws=0.
            for j in range(N):
                ep=_interp1(gg_p,pos[j],vmin,step)
                dp=(gr_v[i]-ep)/gs
                lp=max(np.exp(-0.5*dp*dp) if dp*dp<600. else 0.,1e-300)
                if not np.isnan(gr_sm_v[i]):
                    es=_interp1(gg_s,pos[j],vmin,step)
                    ds=(gr_sm_v[i]-es)/(gs*1.5)
                    ls=max(np.exp(-0.5*ds*ds) if ds*ds<600. else 0.,1e-300)
                    lk=(1.-GR_WT)*lp+GR_WT*ls
                else: lk=lp
                lk=max(lk,1e-300); w[j]*=lk; ws+=w[j]
            if ws>0.:
                for j in range(N): w[j]/=ws
            else:
                for j in range(N): w[j]=1./N
        ws2=0.
        for j in range(N):
            dv=(vel[j]-ve)/max(zsig*2.,0.005)
            lz=max(np.exp(-0.5*dv*dv) if dv*dv<600. else 0.,1e-300)
            w[j]*=lz; ws2+=w[j]
        if ws2>0.:
            for j in range(N): w[j]/=ws2
        else:
            for j in range(N): w[j]=1./N
        ne=0.
        for j in range(N): ne+=w[j]*w[j]
        if 1./ne<RESAMP*N:
            pos,vel=_resamp(pos,vel,w,N,RP,RV)
            for j in range(N): w[j]=1./N
        wm=0.
        for j in range(N): wm+=w[j]*pos[j]
        pts[i]=wm; va=0.
        for j in range(N): va+=w[j]*(pos[j]-wm)**2
        std_[i]=va**0.5; pm=md_v[i]; pz=z_v[i]
    return pts,std_

# Dense grid for O(1) typewell lookup
def _grid(tw_tvt,tw_gr,step=0.2):
    tmin=float(tw_tvt.min()); tmax=float(tw_tvt.max())
    tvt_g=np.arange(tmin,tmax+step,step)
    return np.interp(tvt_g,tw_tvt,tw_gr).astype(np.float64),float(tmin),float(step)

def _gr_sig(hw,tw_tvt,tw_gr):
    kn=hw[hw['TVT_input'].notna()&hw['GR'].notna()]
    if len(kn)<20: return float(PF_GR_SIG_DEF)
    return float(np.clip(np.std(kn['GR'].values-np.interp(kn['TVT_input'].values,tw_tvt,tw_gr)),
                          PF_GR_SIG_MIN,PF_GR_SIG_MAX))

def _nn(arr,v):
    i=int(np.searchsorted(arr,v,'left'))
    if i>=len(arr): return len(arr)-1
    if i>0 and abs(arr[i-1]-v)<=abs(arr[i]-v): return i-1
    return i

def _smooth(vals,fb,r):
    s=pd.Series(vals,dtype='float32').interpolate(limit_direction='both').fillna(fb)
    return (s.rolling(r*2+1,center=True,min_periods=1).mean() if r>0 else s).to_numpy(np.float32)

def beam_search(gr_h,tw_tvt,tw_gr,start_tvt,bs,mc,es,r):
    si=_nn(tw_tvt,start_tvt)
    sgr=_smooth(gr_h,float(np.nanmean(tw_gr)),r).astype(np.float64)
    path=_beam_jit(sgr,tw_gr.astype(np.float64),si,bs,float(mc),float(es))
    return tw_tvt[path].astype(np.float32)

def run_pf_ancc(hw,tw_tvt,tw_gr,N=ANCC_N):
    gs=_gr_sig(hw,tw_tvt,tw_gr)
    kn=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    if len(ev)==0: return np.array([]),np.array([])
    ls=float(kn['TVT_input'].iloc[-1]+kn['Z'].iloc[-1])
    tail=kn.tail(30); dt=np.diff(tail['TVT_input'].values)
    dz=np.diff(tail['Z'].values); dm=np.diff(tail['MD'].values); m=dm>0
    ir=float(np.median((dt+dz)[m]/dm[m])) if m.sum()>=3 else 0.
    gg,gmin,gst=_grid(tw_tvt,tw_gr)
    pts,std=_pf_ancc(ev['MD'].values.astype(np.float64),ev['Z'].values.astype(np.float64),
                      ev['GR'].values.astype(np.float64),gg,gmin,gst,
                      gs,ls,ir,N,ANCC_ALPHA,ANCC_RN,ANCC_PN,ANCC_IS,ANCC_RP,ANCC_RR,PF_RESAMP)
    return pts.astype(np.float32),std.astype(np.float32)

def run_pf_z(hw,tw_tvt,tw_gr,N=PF_N):
    gs=_gr_sig(hw,tw_tvt,tw_gr)
    tw_s=pd.Series(tw_gr).rolling(PF_GR_WIN,center=True,min_periods=1).mean().values.astype(np.float32)
    kna=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    if len(ev)==0: return np.array([]),np.array([])
    dz_k=np.diff(kna['Z'].values); dvt=np.diff(kna['TVT_input'].values)
    dmd_k=np.diff(kna['MD'].values); m2=dmd_k>0
    if m2.sum()>=10:
        vz=dz_k[m2]/dmd_k[m2]; vt=dvt[m2]/dmd_k[m2]
        A=np.column_stack([vz,np.ones_like(vz)]); c,_,_,_=np.linalg.lstsq(A,vt,rcond=None)
        beta,icpt,zsig=float(c[0]),float(c[1]),max(float(np.std(vt-(c[0]*vz+c[1]))),0.001)
    else: beta,icpt,zsig=-1.,0.,0.1
    t2=kna.tail(20); dvt2=np.diff(t2['TVT_input'].values); dmd2=np.diff(t2['MD'].values); m3=dmd2>0
    iv=float(np.median(dvt2[m3]/dmd2[m3])) if m3.sum()>=3 else 0.
    gg,gmin,gst=_grid(tw_tvt,tw_gr)
    gs2,_,_=_grid(tw_tvt,tw_s)
    gr_sm=hw['GR'].rolling(PF_GR_WIN,center=True,min_periods=1).mean()
    pts,std=_pf_z(ev['MD'].values.astype(np.float64),ev['Z'].values.astype(np.float64),
                   ev['GR'].values.astype(np.float64),
                   gr_sm.loc[ev.index].values.astype(np.float64),
                   gg,gs2,gmin,gst,gs,float(kna['TVT_input'].iloc[-1]),iv,
                   beta,icpt,zsig,N,
                   PF_MOM,PF_VN,PF_PN,PF_GR_WT,PF_ROUGH_P,PF_ROUGH_V,PF_RESAMP)
    return pts.astype(np.float32),std.astype(np.float32)


_md=np.linspace(1,50,20,np.float64); _z=np.zeros(20,np.float64); _gr=np.full(20,50.,np.float64)
_gg=np.linspace(45,55,100,np.float64)
_pf_ancc(_md,_z,_gr,_gg,45.,0.1,20.,50.,0.,8,0.998,0.002,0.005,0.3,0.1,0.001,0.5)
_pf_z(_md,_z,_gr,_gr,_gg,_gg,45.,0.1,20.,50.,0.,-1.,0.,0.1,8,0.993,0.005,0.01,0.3,0.2,0.003,0.5)
_beam_jit(np.random.randn(30),np.random.randn(50),25,8,15.,100.)

def robust_slope(x,y,w=None):
    x=np.asarray(x,float); y=np.asarray(y,float)
    m=np.isfinite(x)&np.isfinite(y)
    if m.sum()<2 or np.std(x[m])<1e-6: return 0.
    return float(np.polyfit(x[m],y[m],1)[0])

def affine_cal(kgr,tw_at_k,min_pts=20):
    v=np.isfinite(kgr)&np.isfinite(tw_at_k)
    if v.sum()<min_pts or np.std(tw_at_k[v])<1e-6:
        return 1.,float(np.nanmean(kgr)-np.nanmean(tw_at_k)) if v.any() else 0.
    a,b=np.polyfit(tw_at_k[v],kgr[v],1); return float(a),float(b)

def seg_b_well(ktvt,kz,form_col):
    """Segment b_well: early/mid/late thirds + full prefix.
    Returns (b_full, b_early, b_mid, b_late, b_wls) for feature richness."""
    bv=ktvt+kz-form_col; n=len(bv)
    b_full=float(np.median(bv))
    b_late=float(np.median(bv[max(0,n-50):])) if n>=5 else b_full
    t1,t2=n//3, 2*n//3
    b_early=float(np.median(bv[:max(1,t1)])) if t1>0 else b_full
    b_mid  =float(np.median(bv[t1:max(t1+1,t2)])) if t2>t1 else b_full
    # WLS (tail-upweighted)
    w=np.exp(0.02*np.arange(n)); w/=w.sum()
    b_wls=float(np.dot(w,bv))
    return b_full,b_early,b_mid,b_late,b_wls

def multi_scale_ncc(kgr,ktvt,hgr,hws=(8,15,25),stride=3):
    """Multi-scale NCC. Returns score-weighted ensemble + per-scale signals."""
    out=[]
    for hw in hws:
        win=2*hw+1; nk=len(kgr); nh=len(hgr)
        if nk<win+1 or nh==0:
            out.append((np.full(nh,ktvt[-1],np.float32),np.zeros(nh,np.float32))); continue
        kg=pd.Series(kgr).rolling(5,center=True,min_periods=1).mean().values.astype(np.float32)
        hg=pd.Series(hgr).rolling(5,center=True,min_periods=1).mean().values.astype(np.float32)
        sts=np.arange(0,nk-win+1,stride,dtype=np.int32); M=len(sts)
        if M==0:
            out.append((np.full(nh,ktvt[-1],np.float32),np.zeros(nh,np.float32))); continue
        C=kg[sts[:,None]+np.arange(win,dtype=np.int32)[None,:]].astype(np.float32)
        Cn=(C-C.mean(1,keepdims=True))/(C.std(1,keepdims=True)+1e-6)
        hp=np.pad(hg,hw,mode='edge')
        H=hp[np.arange(nh)[:,None]+np.arange(win)[None,:]].astype(np.float32)
        Hn=(H-H.mean(1,keepdims=True))/(H.std(1,keepdims=True)+1e-6)
        ncc=Hn@Cn.T/win; best=ncc.argmax(1); score=ncc.max(1).astype(np.float32)
        out.append((ktvt[np.clip(sts[best]+hw,0,nk-1)].astype(np.float32),score))
    # Score-weighted ensemble (NEW: softmax-weighted combination)
    tvts=np.stack([o[0] for o in out],1); scores=np.stack([o[1] for o in out],1)
    sw=np.exp(3.*scores); sw/=sw.sum(1,keepdims=True)+1e-9
    sc_ens=(tvts*sw).sum(1).astype(np.float32)
    return out, sc_ens   # [(tvt8,sc8),(tvt15,sc15),(tvt25,sc25)], ensemble

ANCH_OFFS=np.array([-80,-40,-20,-10,-5,0,5,10,20,40,80],np.float32)
BEAM_OFFS=np.array([-40,-20,-10,-5,-3,0,3,5,10,20,40],np.float32)
SC_OFFS  =np.array([-30,-15,-8,-4,-2,0,2,4,8,15,30],np.float32)
PF_OFFS  =np.array([-30,-15,-8,-4,-2,0,2,4,8,15,30],np.float32)

def build_well(hw_path,tw_path,is_train):
    wid=Path(hw_path).stem.replace('__horizontal_well','')
    hw=pd.read_csv(hw_path,usecols=_SAFE_HORIZONTAL_OBSERVABLES)
    tw=pd.read_csv(tw_path,usecols=_SAFE_TYPEWELL_OBSERVABLES).sort_values('TVT')
    kn=hw[hw['TVT_input'].notna()]; ev=hw[hw['TVT_input'].isna()]
    if len(ev)==0 or len(kn)<10: return None
    tw_tvt=tw['TVT'].to_numpy(np.float32); tw_gr=tw['GR'].to_numpy(np.float32)
    if len(tw_tvt)<3: return None

    _seed_fast_safe_feature_rng(_stable_well_seed(_FEATURE_BASE_SEED,wid))
    pf_a,std_a=run_pf_ancc(hw,tw_tvt,tw_gr)
    if len(pf_a)==0: return None
    pf_z,std_z=run_pf_z(hw,tw_tvt,tw_gr)
    pf_use=pf_a.astype(np.float32); std_use=std_a.astype(np.float32)
    has_z=len(pf_z)==len(pf_a) and not np.any(np.isnan(pf_z))

    lk=kn.iloc[-1]; last_tvt=float(lk['TVT_input'])
    gr_full=hw['GR'].astype(float).interpolate(limit_direction='both').fillna(float(np.nanmean(tw_gr)))
    hgr=gr_full.iloc[ev.index[0]:].to_numpy(np.float32)
    kgr=gr_full.iloc[:len(kn)].to_numpy(np.float32)

    # 7 beams (Numba JIT Ã‚Â±2)
    bpaths={}
    for (bs,mc,es,r,tag) in BEAMS:
        bpaths[tag]=beam_search(hgr,tw_tvt,tw_gr,last_tvt,bs,mc,es,r)
    beam_ref=(bpaths['cons']+bpaths['sm5'])/2.

    # Multi-scale NCC Ã¢â€ â€™ score-weighted ensemble
    ktvt=kn['TVT_input'].to_numpy(np.float32)
    sc_res,sc_ens=multi_scale_ncc(kgr,ktvt,hgr,hws=(8,15,25),stride=3)
    sc8,sc8s=sc_res[0]; sc15,sc15s=sc_res[1]; sc25,sc25s=sc_res[2]
    sc_cons=(sc8+sc15+sc25)/3.
    sc_trust=float(np.clip(len(kn)/200.,0.,0.6))
    hyb_ref=(1-sc_trust)*beam_ref+sc_trust*sc_ens  # use ensemble not single

    tw_at_k=np.interp(ktvt,tw_tvt,tw_gr).astype(np.float32)
    a_cal,b_cal=affine_cal(kgr,tw_at_k)
    kmd=kn['MD'].to_numpy(np.float32); kz=kn['Z'].to_numpy(np.float32)
    pfx_rmse=float(np.sqrt(np.mean((kgr-tw_at_k)**2)))
    slp_all=robust_slope(kmd,ktvt); slp_50=robust_slope(kmd[-50:],ktvt[-50:])
    slp_z=robust_slope(kz,ktvt)

    z_kn=kn['Z'].to_numpy(np.float32); z_ev=ev['Z'].to_numpy(np.float32)
    gr_s=pd.Series(gr_full.values); rolls={}
    for w in [5,21,51,101]:
        r=gr_s.rolling(w,center=True,min_periods=1)
        rolls[f'grm{w}']=r.mean().iloc[ev.index].values.astype(np.float32)
        rolls[f'grs{w}']=r.std().fillna(0).iloc[ev.index].values.astype(np.float32)
    for lag in [1,5,15,30]:
        rolls[f'glag{lag}']=gr_s.shift(lag).bfill().iloc[ev.index].values.astype(np.float32)
        rolls[f'glead{lag}']=gr_s.shift(-lag).ffill().iloc[ev.index].values.astype(np.float32)
    gr_d1=gr_s.diff().fillna(0.).iloc[ev.index].values.astype(np.float32)
    gr_d2=gr_s.diff().diff().fillna(0.).iloc[ev.index].values.astype(np.float32)
    gr_env=gr_s.rolling(21,center=True,min_periods=1).max().iloc[ev.index].values.astype(np.float32)
    gr_nrg=np.sqrt(np.maximum((gr_s**2).rolling(21,center=True,min_periods=1).mean(),0.)
                   ).iloc[ev.index].values.astype(np.float32)

    hmd=ev['MD'].to_numpy(np.float32); md_since=hmd-float(lk['MD'])
    slp_b_all=(last_tvt+slp_all*md_since).astype(np.float32)
    slp_b_50 =(last_tvt+slp_50 *md_since).astype(np.float32)

    mdd=hw['MD'].diff().replace(0,np.nan)
    dzdmd=(hw['Z'].diff()/mdd).iloc[ev.index].values.astype(np.float32)
    dxdmd=(hw['X'].diff()/mdd).iloc[ev.index].values.astype(np.float32)
    dydmd=(hw['Y'].diff()/mdd).iloc[ev.index].values.astype(np.float32)

    nh=len(ev); frac=(np.arange(nh)/max(nh-1,1)).astype(np.float32)
    def sc(v): return np.full(nh,np.float32(v),np.float32)

    feats={
        'well':wid,'id':[f'{wid}_{i}' for i in ev.index],
        'last_known_tvt':sc(last_tvt),
        'pf_ancc':pf_use,'pf_ancc_std':std_use,
        'pf_ancc_delta':(pf_use-last_tvt).astype(np.float32),
        'pf_z':(pf_z.astype(np.float32) if has_z else sc(last_tvt)),
        'pf_z_delta':((pf_z-last_tvt).astype(np.float32) if has_z else sc(0.)),
        'pf_vs_z':((pf_use-pf_z.astype(np.float32)) if has_z else sc(0.)),
        **{f'beam_{t}_d':(p-np.float32(last_tvt)).astype(np.float32) for t,p in bpaths.items()},
        'beam_mean_d':np.stack([(p-last_tvt) for p in bpaths.values()],1).mean(1).astype(np.float32),
        'beam_std_d': np.stack([(p-last_tvt) for p in bpaths.values()],1).std(1).astype(np.float32),
        'beam_med_d': np.median(np.stack([(p-last_tvt) for p in bpaths.values()],1),1).astype(np.float32),
        'sc8_d':(sc8-np.float32(last_tvt)).astype(np.float32),'sc8_sc':sc8s,
        'sc15_d':(sc15-np.float32(last_tvt)).astype(np.float32),'sc15_sc':sc15s,
        'sc25_d':(sc25-np.float32(last_tvt)).astype(np.float32),'sc25_sc':sc25s,
        'sc_cons_d':(sc_cons-np.float32(last_tvt)).astype(np.float32),
        'sc_ens_d':(sc_ens-np.float32(last_tvt)).astype(np.float32),
        'sc_trust':sc(sc_trust),'hyb_d':(hyb_ref-np.float32(last_tvt)).astype(np.float32),
        'sc_vs_beam':(sc_ens-bpaths['cons']).astype(np.float32),
        'cal_a':sc(a_cal),'cal_b':sc(b_cal),
        'pfx_rmse':sc(pfx_rmse),'known_len':sc(len(kn)),'eval_len':sc(nh),
        'slp_all':sc(slp_all),'slp_50':sc(slp_50),'slp_z':sc(slp_z),
        'slp_b_d_all':(slp_b_all-last_tvt).astype(np.float32),
        'slp_b_d_50': (slp_b_50-last_tvt).astype(np.float32),
        'ktvt_range':sc(float(np.ptp(ktvt))),'ktvt_std':sc(float(ktvt.std())),
        'md_since':md_since,'frac':frac,'frac2':frac**2,'sqrt_frac':np.sqrt(frac),
        'z':z_ev,
        'dx':(ev['X']-float(lk['X'])).to_numpy(np.float32),
        'dy':(ev['Y']-float(lk['Y'])).to_numpy(np.float32),
        'dz':(z_ev-float(lk['Z'])).astype(np.float32),
        'dxy':np.sqrt((ev['X']-float(lk['X']))**2+(ev['Y']-float(lk['Y']))**2).to_numpy(np.float32),
        'dzdmd':dzdmd,'dxdmd':dxdmd,'dydmd':dydmd,
        'gr':hgr,'gr_d1':gr_d1,'gr_d2':gr_d2,'gr_env':gr_env,'gr_nrg':gr_nrg,
        'gr_vs_tw_anc':hgr-np.float32(np.interp(last_tvt,tw_tvt,tw_gr)),
        'gr_vs_slp_all':hgr-np.interp(slp_b_all,tw_tvt,tw_gr).astype(np.float32),
        **{f'tda{int(o)}' :hgr-np.float32(np.interp(last_tvt+o,tw_tvt,tw_gr)) for o in ANCH_OFFS},
        **{f'tdbc{int(o)}':hgr-np.interp(beam_ref+o,tw_tvt,tw_gr).astype(np.float32) for o in BEAM_OFFS},
        **{f'tdsc{int(o)}':hgr-np.interp(sc_ens+o,tw_tvt,tw_gr).astype(np.float32) for o in SC_OFFS},
        **{f'tdpf{int(o)}':hgr-np.interp(pf_use+o,tw_tvt,tw_gr).astype(np.float32) for o in PF_OFFS},
        'tw_range':sc(float(np.ptp(tw_tvt))),'tw_gr_mean':sc(float(tw_gr.mean())),
    }
    for k,v in rolls.items(): feats[k]=v
    if set(feats)-{'well','id'} != SAFE_FEATURES:
        missing=sorted(SAFE_FEATURES-(set(feats)-{'well','id'}))
        extra=sorted((set(feats)-{'well','id'})-SAFE_FEATURES)
        raise RuntimeError(f'SAFE-only builder contract failed: missing={missing}, extra={extra}')
    return pd.DataFrame(feats)

def build_dataset(paths,is_train,label):
    args=[(str(p),str(p.parent/f'{p.stem.replace("__horizontal_well","")}__typewell.csv'),is_train)
          for p in paths
          if (p.parent/f'{p.stem.replace("__horizontal_well","")}__typewell.csv').exists()]
    t0=time.time()
    res=Parallel(n_jobs=NCPU,prefer='threads',verbose=3)(
        delayed(build_well)(hp,tp,it) for hp,tp,it in args)
    parts=[r for r in res if r is not None]
    return pd.concat(parts,ignore_index=True) if parts else pd.DataFrame()
# <END FROZEN HISTORICAL SAFE SOURCE>
def verify_historical_source_identity() -> bool:
    """Verify that the vendored numerical body still matches its audit hash."""
    source = Path(__file__).read_text(encoding="utf-8")
    begin = "# <" + "BEGIN FROZEN HISTORICAL SAFE SOURCE>\n"
    end = "\n# <" + "END FROZEN HISTORICAL SAFE SOURCE>"
    if source.count(begin) != 1 or source.count(end) != 1:
        raise RuntimeError("frozen historical source markers are missing or ambiguous")
    body = source.split(begin, 1)[1].split(end, 1)[0]
    observed = hashlib.sha256(body.encode("utf-8")).hexdigest()
    if observed != HISTORICAL_SAFE_SOURCE_SHA256:
        raise RuntimeError(
            "vendored historical source changed: "
            f"{observed} != {HISTORICAL_SAFE_SOURCE_SHA256}"
        )
    return True


def _preflight_paths(horizontal_path: Path, typewell_path: Path) -> None:
    horizontal_header = set(pd.read_csv(horizontal_path, nrows=0).columns)
    missing = set(_SAFE_HORIZONTAL_OBSERVABLES) - horizontal_header
    if missing:
        raise ValueError(f"horizontal file is missing observables: {sorted(missing)}")
    typewell_header = set(pd.read_csv(typewell_path, nrows=0).columns)
    missing = set(_SAFE_TYPEWELL_OBSERVABLES) - typewell_header
    if missing:
        raise ValueError(f"type-well file is missing observables: {sorted(missing)}")

    # Read only the allow-listed observables.  A horizontal TVT column may be
    # present in a training CSV, but neither preflight nor the builder loads it.
    horizontal = pd.read_csv(horizontal_path, usecols=_SAFE_HORIZONTAL_OBSERVABLES)
    visible = pd.to_numeric(horizontal["TVT_input"], errors="coerce").notna().to_numpy()
    suffix = np.flatnonzero(~visible)
    if len(suffix) == 0 or int(suffix[0]) < 10:
        raise ValueError("historical builder requires >=10 visible rows and a hidden suffix")
    if not visible[: int(suffix[0])].all() or visible[int(suffix[0]) :].any():
        raise ValueError("TVT_input must be a contiguous visible prefix and hidden suffix")


def build_historical_fast_safe_frame(
    horizontal_path: str | Path,
    typewell_path: str | Path,
) -> pd.DataFrame:
    """Return ``well, id`` and the exact historical 121 SAFE columns.

    The base seed is intentionally frozen to 42.  Changing it would define a
    new stochastic feature artifact rather than reproduce the historical one.
    """
    horizontal_path = Path(horizontal_path)
    typewell_path = Path(typewell_path)
    _preflight_paths(horizontal_path, typewell_path)
    frame = build_well(str(horizontal_path), str(typewell_path), False)
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise RuntimeError(f"historical feature builder returned no rows for {horizontal_path}")
    expected = {"well", "id", *HISTORICAL_FEATURE_ORDER}
    if set(frame.columns) != expected:
        raise RuntimeError(
            "historical feature schema changed: "
            f"missing={sorted(expected-set(frame.columns))}, "
            f"extra={sorted(set(frame.columns)-expected)}"
        )
    ordered = frame.loc[:, ["well", "id", *HISTORICAL_FEATURE_ORDER]].reset_index(drop=True)
    values = ordered.loc[:, HISTORICAL_FEATURE_ORDER].to_numpy(dtype=np.float32)
    if values.shape[1] != 121 or not np.isfinite(values).all():
        raise RuntimeError("historical feature matrix failed its 121-column finite contract")
    return ordered


def build_historical_fast_safe_features(
    horizontal_path: str | Path,
    typewell_path: str | Path,
) -> pd.DataFrame:
    """Return only the sorted 121-column historical model matrix."""
    return build_historical_fast_safe_frame(horizontal_path, typewell_path).loc[
        :, HISTORICAL_FEATURE_ORDER
    ]


def canonical_historical_feature_hash(frame: pd.DataFrame) -> str:
    """Hash IDs and little-endian float32 feature bytes in canonical order."""
    required = {"id", *HISTORICAL_FEATURE_ORDER}
    if not required.issubset(frame.columns):
        raise ValueError(f"frame is missing columns: {sorted(required-set(frame.columns))}")
    digest = hashlib.sha256()
    for value in frame["id"].astype(str):
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(4, "little"))
        digest.update(encoded)
    values = np.ascontiguousarray(
        frame.loc[:, HISTORICAL_FEATURE_ORDER].to_numpy(dtype="<f4")
    )
    digest.update(values.tobytes())
    return digest.hexdigest()
