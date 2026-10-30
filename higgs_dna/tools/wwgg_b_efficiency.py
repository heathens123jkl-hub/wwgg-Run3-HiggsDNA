"""Pre-b-veto sufficient statistics, not efficiency estimates or SF weights."""
import awkward as ak
import numpy as np

PT_EDGES = [30.,50.,70.,100.,140.,200.,300.,600.,1000.,float('inf')]
ETA_EDGES = [0.,.8,1.6,2.5]


def collect(jets, genweight, category):
    required={'pt','eta','btagUParTAK4B','hadronFlavour'}
    if required-set(jets.fields):raise ValueError('Missing efficiency fields: '+repr(required-set(jets.fields)))
    if len(jets)!=len(genweight) or len(jets)!=len(category):raise ValueError('Unaligned events')
    if not bool(ak.all(np.isfinite(genweight))):raise ValueError('Nonfinite generator weights')
    if not bool(ak.all((category>=0)&(category<=2))):raise ValueError('Invalid category')
    if not bool(ak.all(np.isfinite(jets.pt)&np.isfinite(jets.eta))):raise ValueError('Nonfinite jet kinematics')
    eligible=(jets.pt>30)&(abs(jets.eta)<2.5)
    selected=jets[eligible]
    flat=lambda x:np.asarray(ak.to_numpy(ak.flatten(x)))
    pt,eta,score,flavour=[flat(selected[f]) for f in ('pt','eta','btagUParTAK4B','hadronFlavour')]
    if not np.all(np.isfinite(score)&(score>=0)&(score<=1)&np.isin(flavour,[0,4,5])):
        raise ValueError('Invalid accepted score/flavour')
    weights=flat(ak.broadcast_arrays(selected.pt,genweight)[1])
    cats=flat(ak.broadcast_arrays(selected.pt,category)[1])
    tagged=score>=.4648
    result={'schema':'wwgg_b_efficiency_v1','threshold':.4648,
            'pt_edges':[str(x) if np.isinf(x) else x for x in PT_EDGES],
            'abs_eta_edges':ETA_EDGES,'weight_convention':'genWeight only; before b veto, after configured tau veto',
            'events_before_b_veto':len(jets),'channels':{}}
    for cat in (0,1,2):
        by_flavour={};result['channels'][str(cat)]=by_flavour
        for f in (0,4,5):
            moments={};by_flavour[str(f)]=moments
            for label,mask in [('all',(cats==cat)&(flavour==f)),('tagged',(cats==cat)&(flavour==f)&tagged)]:
                moments[label]={name:np.histogram2d(pt[mask],abs(eta[mask]),bins=[PT_EDGES,ETA_EDGES],weights=w[mask])[0].tolist()
                    for name,w in [('rows',np.ones(len(pt))),('sumw',weights),('sumw2',weights**2),
                                   ('sumabsw',abs(weights)),('negative_rows',(weights<0).astype(float))]}
    return result
