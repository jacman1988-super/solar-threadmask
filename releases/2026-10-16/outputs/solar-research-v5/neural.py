"""Compact supervised U-Net, trained from scratch on local mask annotations."""
import argparse,json,time,hashlib,sys
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from research import Dataset,HERE,CACHE,channels,write,sha

class Block(nn.Sequential):
    def __init__(self,a,b):
        super().__init__(nn.Conv2d(a,b,3,padding=1,bias=False),nn.GroupNorm(3,b),nn.SiLU(),nn.Conv2d(b,b,3,padding=1,bias=False),nn.GroupNorm(3,b),nn.SiLU())

class Net(nn.Module):
    def __init__(self):
        super().__init__();self.a=Block(3,12);self.b=Block(12,24);self.c=Block(24,48);self.d=Block(48,96)
        self.u=Block(144,48);self.v=Block(72,24);self.w=Block(36,12);self.out=nn.Conv2d(12,1,1)
    def forward(self,x):
        a=self.a(x);b=self.b(F.avg_pool2d(a,2));c=self.c(F.avg_pool2d(b,2));d=self.d(F.avg_pool2d(c,2))
        u=self.u(torch.cat([F.interpolate(d,size=c.shape[-2:],mode='bilinear',align_corners=False),c],1))
        v=self.v(torch.cat([F.interpolate(u,size=b.shape[-2:],mode='bilinear',align_corners=False),b],1))
        return self.out(self.w(torch.cat([F.interpolate(v,size=a.shape[-2:],mode='bilinear',align_corners=False),a],1)))

def train():
    torch.set_num_threads(4);torch.manual_seed(20261015);rng=np.random.default_rng(20261015)
    # Separate raw-array cache permits concurrent RF preprocessing without shared writes.
    import research
    research.CACHE=CACHE/'neural-arrays'
    d=Dataset();names=d.protocol['split']['train'];manifest={'protocol_sha256':sha(HERE/'protocol.json'),'source_sha256':sha(__file__),'torch':torch.__version__,'parameters':sum(p.numel() for p in Net().parameters()),'steps':900,'batch':6,'patch':192,'seed':20261015,'checkpoints':[300,600,900]}
    if (HERE/'neural-training.json').exists():raise FileExistsError('Preserve prior run')
    write(HERE/'neural-training.json',manifest)
    print('Preparing neural inputs',flush=True)
    positives={}
    for k,name in enumerate(names,1):
        im,target,disk=d.arrays(name)
        p=CACHE/(name+'.channels.npy');q=CACHE/(name+'.target.npy')
        if not p.exists():np.save(p,channels(im,disk).astype(np.float16))
        if not q.exists():np.save(q,target.astype(np.float16))
        positives[name]=np.argwhere(target>.25)[::4]
        print('Prepared',k,flush=True)
    model=Net();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=1e-4)
    logs=[];start=time.perf_counter()
    for step in range(1,901):
        xs=[];ys=[]
        for _ in range(6):
            name=names[int(rng.integers(len(names)))];pos=positives[name]
            if len(pos) and rng.random()<.7:
                cy,cx=pos[int(rng.integers(len(pos)))];cy+=int(rng.integers(-64,65));cx+=int(rng.integers(-64,65))
            else:cy,cx=rng.integers(120,904,size=2)
            y,x=int(np.clip(cy-96,0,832)),int(np.clip(cx-96,0,832))
            a=np.load(CACHE/(name+'.channels.npy'),mmap_mode='r')[:,y:y+192,x:x+192].astype(np.float32)
            b=np.load(CACHE/(name+'.target.npy'),mmap_mode='r')[None,y:y+192,x:x+192].astype(np.float32)
            rot=int(rng.integers(4));a=np.rot90(a,rot,axes=(1,2));b=np.rot90(b,rot,axes=(1,2))
            if rng.random()<.5:a=a[:,:,::-1];b=b[:,:,::-1]
            xs.append(a.copy());ys.append(b.copy())
        xx=torch.from_numpy(np.stack(xs));yy=torch.from_numpy(np.stack(ys));pred=model(xx)
        prob=pred.sigmoid();dice=1-(2*(prob*yy).sum()+1)/(prob.sum()+yy.sum()+1)
        loss=F.binary_cross_entropy_with_logits(pred,yy)+dice
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5);opt.step()
        if step%25==0:
            row={'step':step,'loss':float(loss.detach()),'seconds':time.perf_counter()-start};logs.append(row);print(row,flush=True)
        if step in [300,600,900]:torch.save(model.state_dict(),HERE/f'unet{step}.pt')
    write(HERE/'neural-loss.json',logs)

def predict(ch,path):
    torch.set_num_threads(4);model=Net();model.load_state_dict(torch.load(path,map_location='cpu',weights_only=True));model.eval()
    # Overlap-add 256 tiles; discard context margin by smooth positive blending.
    h,w=ch.shape[-2:];out=np.zeros((h,w),np.float32);weight=np.zeros_like(out)
    positions=sorted(set(list(range(0,h-256,192))+[h-256]))
    axis=np.maximum(np.hanning(256),.05);blend=np.outer(axis,axis).astype(np.float32)
    with torch.no_grad():
        for y in positions:
            for x in positions:
                p=model(torch.from_numpy(ch[None,:,y:y+256,x:x+256])).sigmoid()[0,0].numpy()
                out[y:y+256,x:x+256]+=p*blend;weight[y:y+256,x:x+256]+=blend
    return out/np.maximum(weight,1e-6)

if __name__=='__main__':train()
