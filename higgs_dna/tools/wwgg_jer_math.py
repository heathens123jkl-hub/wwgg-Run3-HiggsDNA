import numpy as np

def match(pt,eta,phi,resolution,gen):
    if not gen['GenJet_pt']:
        return -1.,False
    gp=np.asarray(gen['GenJet_pt']);ge=np.asarray(gen['GenJet_eta']);gf=np.asarray(gen['GenJet_phi'])
    dp=np.arctan2(np.sin(phi-gf),np.cos(phi-gf))
    dr=np.hypot(eta-ge,dp)
    good=(dr<.2)&(np.abs(pt-gp)<3*resolution*pt)&(gp>0)
    if not good.any():return -1.,False
    k=np.argmin(np.where(good,dr,np.inf))
    return float(gp[k]),True

def factors(pt,gp,matched,res,sf,z):
    return np.where(matched,1+(sf-1)*(pt-gp)/pt,
                    1+z*res*np.sqrt(np.maximum(sf*sf-1,0)))
