"""Small helpers for one-subsensor master/slave transfer."""
import json
from pathlib import Path
import numpy as np
from sklearn.metrics import r2_score
from day4_utils import predict_normalized

def group_values(y, pred, groups):
    _, inv = np.unique(groups, return_inverse=True); n = np.bincount(inv)
    return (np.bincount(inv, weights=np.asarray(y).ravel())/n,
            np.bincount(inv, weights=np.asarray(pred).ravel())/n)

def metrics(y, pred, groups):
    yt, yp = group_values(y, pred, groups)
    return {"UGM_RMSE_ppb":float(np.sqrt(np.mean((yt-yp)**2))),
            "UGM_R2":float(r2_score(yt,yp)),"UGMs":int(len(yt))}

def copy_state(model):
    return {k:v.detach().cpu().clone() for k,v in model.model.state_dict().items()}

def aligned_master_slave_pairs(domains, masters, slave_subset, split="train"):
    """Use every individual master as a regression target; no master mean."""
    rows=slave_subset[split]["rows"]; groups=slave_subset[split]["groups"]
    slave=slave_subset[split]["X_z"]; xs=[]; ys=[]; ids=[]
    for sensor in masters:
        source=domains[sensor][split]
        lookup={int(row):i for i,row in enumerate(source["rows"])}
        index=np.asarray([lookup[int(row)] for row in rows])
        assert np.array_equal(source["groups"][index],groups)
        xs.append(slave); ys.append(source["X_z"][index]); ids += [sensor]*len(slave)
    x,y=np.concatenate(xs),np.concatenate(ys)
    assert x.shape==y.shape and x.shape[1:]==(1,1440,1)
    return x,y,np.asarray(ids)

class DirectStandardization:
    def __init__(self,alpha=.1): self.alpha=float(alpha)
    def fit(self,slave,master):
        self.shape=slave.shape[1:]; x=slave.reshape(len(slave),-1).astype(float)
        y=master.reshape(len(master),-1).astype(float)
        self.xm,self.ym=x.mean(0),y.mean(0); self.x=x-self.xm
        self.dual=np.linalg.solve(self.x@self.x.T+self.alpha*np.eye(len(x)),y-self.ym)
        return self
    def transform(self,values):
        x=values.reshape(len(values),-1)-self.xm
        return (x@self.x.T@self.dual+self.ym).reshape((len(values),)+self.shape).astype("float32")

class PiecewiseDirectStandardization:
    def __init__(self,alpha=.1,radius=2): self.alpha,self.radius=float(alpha),int(radius)
    def windows(self,values):
        x=np.asarray(values,float)[...,0]
        x=np.pad(x,((0,0),(0,0),(self.radius,self.radius)),mode="edge")
        return np.lib.stride_tricks.sliding_window_view(x,2*self.radius+1,axis=2)
    def fit(self,slave,master):
        x=self.windows(slave); y=np.asarray(master,float)[...,0]
        self.xm,self.ym=x.mean(0),y.mean(0); x,y=x-self.xm,y-self.ym
        gram=np.einsum("nctw,nctv->ctwv",x,x,optimize=True)
        rhs=np.einsum("nctw,nct->ctw",x,y,optimize=True)
        gram += self.alpha*np.eye(x.shape[-1])[None,None]
        self.coef=np.linalg.solve(gram,rhs[...,None])[...,0]; return self
    def transform(self,values):
        y=np.einsum("nctw,ctw->nct",self.windows(values)-self.xm,self.coef,optimize=True)
        return (y+self.ym)[...,None].astype("float32")

def mapped_predictions(model,adapter,split,scaler):
    z=predict_normalized(model,adapter.transform(split["X_z"])).ravel()
    return z*scaler["y_std"]+scaler["y_mean"]

def scatter_grid(split,predictions,title):
    import matplotlib.pyplot as plt
    vals={name:group_values(split["y"],p,split["groups"]) for name,p in predictions.items()}
    lo=min(min(a.min(),b.min()) for a,b in vals.values()); hi=max(max(a.max(),b.max()) for a,b in vals.values())
    fig,axes=plt.subplots(1,len(vals),figsize=(4.2*len(vals),4.2),squeeze=False)
    for ax,(name,(a,b)) in zip(axes.ravel(),vals.items()):
        ax.scatter(a,b,s=22,alpha=.7); ax.plot([lo,hi],[lo,hi],"k--")
        ax.set(title=f"{name}\nRMSE={np.sqrt(np.mean((a-b)**2)):.1f} ppb",
               xlabel="True UGM mean [ppb]",ylabel="Predicted UGM mean [ppb]")
        ax.grid(True,alpha=.3)
    fig.suptitle(title); fig.tight_layout(); plt.show()

def save_json(path,data):
    Path(path).write_text(json.dumps(data,indent=2,default=lambda x:x.item() if isinstance(x,np.generic) else str(x)),encoding="utf-8")
