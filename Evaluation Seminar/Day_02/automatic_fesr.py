"""Automatically learned piecewise-linear features for sklearn pipelines."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import RFE, f_regression
from sklearn.linear_model import Ridge
from sklearn.utils.validation import check_is_fitted

class AutomaticSegments(BaseEstimator, TransformerMixin):
    """Greedy BIC-penalized segmentation of the fold's median training cycle.

    Segment count is learned, not supplied. Minimum length is a resolution limit.
    The BIC-like objective is a complexity heuristic, not an exhaustive optimum.
    """
    def __init__(self, min_length=8, sample_rate=10.):
        self.min_length=min_length; self.sample_rate=sample_rate
    def fit(self,X,y=None):
        X=np.asarray(X,dtype=float)
        if X.ndim!=3 or X.shape[1]!=1: raise ValueError('Expected (cycles,1,time).')
        self.n_time_=X.shape[2]; self.reference_=np.median(X[:,0],axis=0)
        ref=self.reference_; n=len(ref); t=np.arange(n,dtype=float)
        if n<2*self.min_length: raise ValueError('Cycle is too short for this resolution.')
        prefix=[np.r_[0,np.cumsum(v)] for v in [ref,ref**2,t,t**2,t*ref]]
        def cost(a,b):
            length=b-a
            sy,syy,st,stt,sty=[v[b]-v[a] for v in prefix]
            return np.maximum(syy-sy**2/length-(sty-st*sy/length)**2/np.maximum(stt-st**2/length,1e-15),0.)
        floor=max(float(np.var(ref)),1.)*1e-12
        def criterion(rss,k): return n*np.log(max(rss/n,floor))+(3*k-1)*np.log(n)
        segments=[(0,n)]; rss=float(cost(0,n)); score=criterion(rss,1)
        self.path_=[(1,score,rss)]
        while True:
            candidate=None
            for i,(a,b) in enumerate(segments):
                cuts=np.arange(a+self.min_length,b-self.min_length+1)
                if not len(cuts): continue
                gains=cost(a,b)-cost(a,cuts)-cost(cuts,b)
                j=int(np.argmax(gains)); item=(float(gains[j]),i,int(cuts[j]))
                if candidate is None or item[0]>candidate[0]: candidate=item
            if candidate is None: break
            gain,i,cut=candidate; proposed=max(rss-gain,0.)
            next_score=criterion(proposed,len(segments)+1)
            if next_score>=score: break
            a,b=segments.pop(i); segments[i:i]=[(a,cut),(cut,b)]
            rss,score=proposed,next_score
            self.path_.append((len(segments),score,rss))
        self.boundaries_=np.array([a for a,b in segments]+[n],dtype=int)
        self.n_segments_=len(segments)
        self.feature_names_=np.array([f'segment{i:03d}_{a:04d}-{b-1:04d}_{kind}'
            for i,(a,b) in enumerate(segments) for kind in ['mean','slope_per_s']],dtype=object)
        return self
    def transform(self,X):
        check_is_fitted(self,'boundaries_'); X=np.asarray(X,dtype=float)
        if X.shape[1:]!=(1,self.n_time_): raise ValueError('Unexpected cycle shape.')
        columns=[]
        for a,b in zip(self.boundaries_[:-1],self.boundaries_[1:]):
            values=X[:,0,a:b]; t=np.arange(b-a)/self.sample_rate; t-=t.mean()
            columns.extend([values.mean(axis=1),values@t/(t@t)])
        return np.column_stack(columns)
    def get_feature_names_out(self,input_features=None):
        check_is_fitted(self,'feature_names_'); return self.feature_names_

class FractionSelector(BaseEstimator, TransformerMixin):
    """Choose a fraction of the fold-dependent feature count; always retain >=2."""
    def __init__(self,method='pearson',fraction=.5): self.method=method; self.fraction=fraction
    def fit(self,X,y):
        count=min(X.shape[1],max(2,int(np.ceil(X.shape[1]*self.fraction))))
        if self.method=='pearson':
            scores,_=f_regression(X,y)
            self.indices_=np.sort(np.argsort(-scores,kind='stable')[:count])
        elif self.method=='rfe':
            selector=RFE(Ridge(alpha=1.),n_features_to_select=count,step=.2).fit(X,y)
            self.indices_=np.flatnonzero(selector.support_)
        else: raise ValueError('Unknown selection method.')
        return self
    def transform(self,X): check_is_fitted(self,'indices_'); return np.asarray(X)[:,self.indices_]
