import pandas as pd, numpy as np, math, os, random, shutil, imageio.v3 as iio
O='/data/yuzhewu/eybxroam/overfit4'
oc=pd.read_csv(f'{O}/other_list.csv')
miss=[r['clip'] for _,r in oc.iterrows() if not os.path.exists(f"{O}/actions_world/{r['clip']}.parquet")]
print("missing relabels:", len(miss), flush=True)
oc=oc[~oc['clip'].isin(set(miss))]
for _,r in oc.iterrows():
    shutil.copy(f"{O}/actions_world/{r['clip']}.parquet", f"{O}/actions/{r['clip']}.parquet")
print("labels installed", len(oc), flush=True)
idx=pd.read_csv(f'{O}/clips_index.csv')
trio=idx[idx.kind=='normal']
tk={}; ok_={}
for c in trio['clip']: tk[c]=pd.read_parquet(f'{O}/actions/{c}.parquet')['keys']
for c in oc['clip']: ok_[c]=pd.read_parquet(f'{O}/actions/{c}.parquet')['keys']
regmap=dict(zip(trio['clip'], trio['region']))
CUTS=[1+4*m for m in range(3,23)]
def pshift(a,b):
    A=np.fft.fft2(a); B=np.fft.fft2(b)
    r=A*np.conj(B); r/=np.abs(r)+1e-9
    cc=np.abs(np.fft.ifft2(r))
    dy,dx=np.unravel_index(cc.argmax(), cc.shape)
    if dy>a.shape[0]//2: dy-=a.shape[0]
    if dx>a.shape[1]//2: dx-=a.shape[1]
    return dx,dy
def seam_ok(fr,c):
    g=fr[...,0:3].mean(-1)[:, 30:450, 180:652]
    def acc(rng):
        vx=vy=0.0
        for t in rng: dx,dy=pshift(g[t],g[t+1]); vx+=dx; vy+=dy
        return math.degrees(math.atan2(vy,vx))%360, math.hypot(vx,vy)
    lo=max(c-8,1); hi=min(c+8,100)
    (a1,m1),(a2,m2)=acc(range(lo,c-1)),acc(range(c+1,hi))
    if m1<5 or m2<5: return False
    d=abs(a1-a2)%360
    return min(d,360-d)<=25
random.seed(31)
target={'infiniteDungeon':84,'mountainPass':83,'hills':83}
byreg={rg:[c for c,r in regmap.items() if r==rg] for rg in target}
oclist=list(ok_.keys())
added=[]; tried=0
while any(v>0 for v in target.values()) and tried<40000:
    tried+=1
    rb=random.choice(list(target))
    if target[rb]<=0: continue
    ca=random.choice(oclist); cb=random.choice(byreg[rb]); c=random.choice(CUTS)
    ka=ok_[ca]; kb=tk[cb]
    if len(ka)<=c or len(kb)<=c: continue
    if ka.iloc[c-1]=='' or ka.iloc[c-1]!=kb.iloc[c]: continue
    cid=f'spn_{ca}_{cb}_{c}'
    if os.path.exists(f'{O}/clips/{cid}.mp4'): continue
    fa=iio.imread(f'{O}/clips/{ca}.mp4'); fb=iio.imread(f'{O}/clips/{cb}.mp4')
    out=np.concatenate([fa[:c], fb[c:101]])
    if not seam_ok(out,c): continue
    iio.imwrite(f'{O}/clips/{cid}.mp4', out, fps=20, codec='libx264', output_params=['-crf','18','-pix_fmt','yuv420p'])
    A=pd.read_parquet(f'{O}/actions/{ca}.parquet').iloc[:c]
    B=pd.read_parquet(f'{O}/actions/{cb}.parquet').iloc[c:101]
    pd.concat([A,B]).reset_index(drop=True).to_parquet(f'{O}/actions/{cid}.parquet')
    added.append(dict(clip=cid, kind='splice_null', region=f'null>{rb}', cut=c))
    target[rb]-=1
    if len(added)%50==0: print(len(added), "built", flush=True)
print("null splices:", len(added), "tried", tried, "remaining", {k:v for k,v in target.items() if v>0}, flush=True)
pd.DataFrame(added).to_csv(f'{O}/splices_null.csv', index=False)
ocn=pd.DataFrame(dict(clip=oc['clip'], kind='normal_other', region=oc['region'], cut=-1))
idx2=pd.concat([idx, ocn, pd.DataFrame(added)])
idx2.to_csv(f'{O}/clips_index.csv', index=False)
SCENES=['infiniteDungeon','mountainPass','hills']
def sid(rg): return float(SCENES.index(rg)) if rg in SCENES else 3.0
rows=[]
for _,r in idx2.iterrows():
    c=r['clip']
    a=pd.read_parquet(f'{O}/actions/{c}.parquet')
    std=pd.DataFrame({
        'W': a['keys'].str.contains('W').astype('float32'),
        'A': a['keys'].str.contains('A').astype('float32'),
        'S': a['keys'].str.contains('S').astype('float32'),
        'D': a['keys'].str.contains('D').astype('float32'),
        'MOUSE0': 0.0, 'CAM_X': 0.0, 'CAM_Y': 0.0, 'CAM_ACTIVE': 0.0,
        'SCENE': a['rg'].map(sid).astype('float32'),
    })
    std.to_parquet(f'{O}/actions_std/{c}.parquet')
    rows.append(dict(video=f'clips/{c}.mp4', action=f'actions_std/{c}.parquet', prompt='',
                     session='eybx', chunk=r['kind'], source_start_sec=0.0,
                     move_frames=int((a['keys']!='').sum()), mouse0_frames=0))
pd.DataFrame(rows).to_csv(f'{O}/metadata.csv', index=False)
print("metadata:", len(rows), dict(idx2.groupby('kind').size()), flush=True)
print("NULLSPLICE_DONE", flush=True)
