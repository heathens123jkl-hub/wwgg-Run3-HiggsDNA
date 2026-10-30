"""Explicit WWgg 2024 integration; independent variation jobs rerun selection.

The bundled configuration is a validation candidate, not a full uncertainty model.
No tau/b-veto efficiency SF or extra prompt-MVA efficiency SF is invented here.
"""
from functools import lru_cache
from pathlib import Path
import hashlib
import json
import awkward as ak
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
PAYLOADS = ROOT / 'higgs_dna/systematics/JSONs/WWgg/2024'


def options(config=None):
    cfg = dict(jme=False, jer=False, jer_policy='cmssw_energy_floor',
               jer_forward_exemption=False, lepton_sf=False, tau_veto='off',
               b_veto='off', variation='nominal', flow_precision='float64',
               tau_config='production/wwgg/tau_veto_2024.json', b_efficiency=False)
    if config:
        if not isinstance(config, dict) or set(config) - set(cfg):
            raise ValueError('Unknown WWgg correction configuration')
        cfg.update(config)
    for k in ('jme', 'jer', 'jer_forward_exemption', 'lepton_sf', 'b_efficiency'):
        if type(cfg[k]) is not bool:
            raise ValueError(k + ' must be boolean')
    for k in ('tau_veto', 'b_veto'):
        if cfg[k] not in ('off', 'store_only', 'apply'):
            raise ValueError('Invalid veto mode: ' + k)
    if cfg['flow_precision'] != 'float64':
        raise ValueError('WWgg validated flow requires float64')
    if cfg['jer_policy'] not in ('zero', 'unity', 'cmssw_energy_floor'):
        raise ValueError('Invalid JER nonpositive-factor policy')
    if cfg['jer'] and not cfg['jme']:
        raise ValueError('JRV2 requires the V5 JEC integration')
    from .wwgg_jec_type1 import SOURCES
    valid = {'nominal'} | {f'{s}_{d}' for s in SOURCES for d in ('up','down')}
    valid |= {f'{s}_{d}' for s in ('JER','mu_id','mu_iso','electron_id','electron_reco') for d in ('up','down')}
    if cfg['variation'] not in valid:
        raise ValueError('Unknown variation ' + cfg['variation'])
    v = cfg['variation']
    if v.startswith('JER_') and not cfg['jer']:
        raise ValueError('JER variation requested without JER')
    if any(v.startswith(s + '_') for s in SOURCES) and not cfg['jme']:
        raise ValueError('JES variation requested without JEC')
    if any(v.startswith(s + '_') for s in ('mu_id','mu_iso','electron_id','electron_reco')) and not cfg['lepton_sf']:
        raise ValueError('Lepton variation requested without lepton SF')
    return cfg


@lru_cache(None)
def payload(name):
    import correctionlib
    return correctionlib.CorrectionSet.from_file(str(PAYLOADS / name))


def provenance(cfg):
    names = []
    if cfg['jme']: names += ['jet_jerc.json.gz', 'jetid.json.gz']
    if cfg['lepton_sf']: names += ['electron.json.gz', 'muon_Z.json.gz']
    result = {n: hashlib.sha256((PAYLOADS/n).read_bytes()).hexdigest() for n in names}
    if cfg['tau_veto'] != 'off':
        result['tau_config'] = hashlib.sha256((ROOT/cfg['tau_config']).read_bytes()).hexdigest()
    if cfg['b_veto'] != 'off':
        from .wwgg_b_veto import load_b_config
        result['btag_wp_payload'] = load_b_config(ROOT)['payload_sha256']
    return result


def jet_id(jets):
    from .wwgg_jec_type1 import flat
    counts = ak.num(jets.pt)
    values = {n: flat(jets[n]) for n in ('eta','chHEF','neHEF','chEmEF','neEmEF','muEF','chMultiplicity','neMultiplicity')}
    values['multiplicity'] = values['chMultiplicity'] + values['neMultiplicity']
    out = np.zeros(int(ak.sum(counts)), dtype=np.int32)
    for key, bit in [('AK4PUPPI_Tight',2),('AK4PUPPI_TightLeptonVeto',4)]:
        c = payload('jetid.json.gz')[key]
        args = [values[i.name].astype(np.int64) if i.type == 'int' else values[i.name] for i in c.inputs]
        if len(out): out |= np.asarray(c.evaluate(*args),dtype=np.int32)*bit
    return ak.unflatten(out,counts)


def apply_jme(events, cfg, is_data):
    """Rebuild Type-1 from raw MET; JER intentionally NOT propagated to MET."""
    from .wwgg_jec_type1 import BRANCHES, calculate, flat
    fields = {}
    for b in BRANCHES:
        if '_' in b:
            collection, field = b.split('_',1)
            fields[b] = events[collection][field]
        else: fields[b] = events[b]
    inputs = ak.zip(fields, depth_limit=1)
    variation = cfg['variation']
    if is_data and variation != 'nominal':
        raise ValueError('MC variations must not be run on collision Data')
    cset = payload('jet_jerc.json.gz')
    result = calculate(inputs,cset,is_data,include_jes=not is_data and variation not in ('nominal','JER_up','JER_down') and not variation.startswith(('mu_','electron_')))
    key = variation if variation in result['jets'] else 'nominal'
    original = events.Jet
    jec_pt, jec_mass = result['jets']['nominal']['pt'], result['jets']['nominal']['mass']
    smear = ak.ones_like(jec_pt)
    if cfg['jer'] and not is_data:
        smear = jer_factors(events,jec_pt,jec_mass,cset,cfg)
    jets = ak.with_field(original, original.pt, 'pt_nano')
    jets = ak.with_field(jets, original.mass, 'mass_nano')
    jets = ak.with_field(jets, original.pt*(1-original.rawFactor), 'pt_raw')
    jets = ak.with_field(jets, jec_pt, 'pt_jec')
    jets = ak.with_field(jets, smear, 'jer_factor')
    jets = ak.with_field(jets,result['jets'][key]['pt']*smear,'pt')
    jets = ak.with_field(jets,result['jets'][key]['mass']*smear,'mass')
    jets = ak.with_field(jets,jet_id(jets),'jetId')
    # The changed pt changes which jet is leading, including for BDT features.
    events['Jet'] = jets[ak.argsort(jets.pt,axis=1,ascending=False,stable=True)]
    met = events.PuppiMET
    met = ak.with_field(met,met.pt,'pt_nano')
    met = ak.with_field(met,met.phi,'phi_nano')
    met = ak.with_field(met,result['met'][key]['pt'],'pt')
    met = ak.with_field(met,result['met'][key]['phi'],'phi')
    events['PuppiMET'] = met
    return events


def jer_factors(events, pts, masses, cset, cfg):
    import correctionlib
    from .wwgg_jec_type1 import flat, evaluate
    from .wwgg_jer_math import match, factors
    counts = ak.to_numpy(ak.num(pts))
    pt, mass = flat(pts), flat(masses)
    eta, phi = flat(events.Jet.eta), flat(events.Jet.phi)
    rho = np.repeat(ak.to_numpy(events.Rho.fixedGridRhoFastjetAll), counts).astype(float)
    if not len(pt): return ak.unflatten(np.array([],dtype=float),counts)
    if np.any(pt<=0): raise ValueError('JER requires positive JEC pt')
    v = dict(JetPt=pt,JetEta=eta,Rho=rho)
    tag = 'Summer24Prompt24_JRV2_MC_'
    res,sf,unc = [evaluate(cset[tag+n+'_AK4PFPuppi'],v) for n in ('PtResolution','ScaleFactor','SFUncertainty')]
    if not np.isfinite([res,sf,unc]).all() or np.any((res<0)|(sf<=0)|(unc<0)|(unc>=1)):
        raise ValueError('JER payload outside validated range')
    genpt, matched = np.empty(len(pt)), np.empty(len(pt),dtype=bool)
    offset = 0
    for n, gen in zip(counts,ak.to_list(events.GenJet)):
        g = {'GenJet_'+k:[j[k] for j in gen] for k in ('pt','eta','phi')}
        for i in range(offset,offset+n): genpt[i],matched[i] = match(pt[i],eta[i],phi[i],res[i],g)
        offset += n
    rng = correctionlib.CorrectionSet.from_string(json.dumps({'schema_version':2,'corrections':[{
        'name':'normal','version':1,'inputs':[{'name':n,'type':t} for n,t in
        [('JetPt','real'),('JetEta','real'),('Rho','real'),('EventID','int')]],
        'output':{'name':'draw','type':'real'},'data':{'nodetype':'hashprng',
        'inputs':['JetPt','JetEta','Rho','EventID'],'distribution':'normal'}}]}))['normal']
    z = rng.evaluate(pt,eta,rho,np.repeat(ak.to_numpy(events.event),counts).astype(np.int64))
    # Same central momenta, matching and random draw in all variation jobs.
    variation = cfg['variation']
    if variation == 'JER_up': sf = sf*(1+unc)
    if variation == 'JER_down': sf = sf*(1-unc)
    f = factors(pt,genpt,matched,res,sf,z)
    if cfg['jer_forward_exemption']: f = np.where(~matched&(abs(eta)>2.5)&(abs(eta)<3),1.,f)
    if not np.isfinite(f).all(): raise ValueError('Nonfinite JER factor')
    if cfg['jer_policy']=='zero': f = np.maximum(0,f)
    elif cfg['jer_policy']=='unity': f = np.where(f>0,f,1.)
    else: f = np.maximum(.01/np.hypot(pt*np.cosh(eta),mass),f)
    return ak.unflatten(f,counts)


def lepton_components(electrons,muons):
    """Component SFs only: not a correction for the extra prompt-MVA cuts."""
    result = {}
    for component in ('mu_id','mu_iso','electron_id','electron_reco'):
        is_mu = component.startswith('mu_')
        obj = muons if is_mu else electrons
        counts = ak.num(obj.pt)
        pt = np.asarray(ak.to_numpy(ak.flatten(obj.pt)),dtype=float)
        eta = np.asarray(ak.to_numpy(ak.flatten(obj.eta if is_mu else obj.eta+obj.deltaEtaSC)),dtype=float)
        if not np.isfinite([pt,eta]).all() or np.any((pt<10)|(abs(eta)>=(2.4 if is_mu else 2.5))):
            raise ValueError('Selected lepton outside SF domain')
        for variation in ('nominal','up','down'):
            values = np.ones(len(pt))
            if is_mu and len(pt):
                key = 'NUM_MediumID_DEN_TrackerMuons' if component=='mu_id' else 'NUM_LoosePFIso_DEN_MediumID'
                values = payload('muon_Z.json.gz')[key].evaluate(eta,pt,{'nominal':'nominal','up':'systup','down':'systdown'}[variation])
            elif len(pt):
                c = payload('electron.json.gz')['Electron-ID-SF']
                parts = [('Loose',np.ones(len(pt),dtype=bool))] if component=='electron_id' else [
                    ('RecoBelow20',pt<20),('Reco20to75',(pt>=20)&(pt<75)),('RecoAbove75',pt>=75)]
                for wp,mask in parts:
                    if mask.any(): values[mask] = c.evaluate('2024Prompt',{'nominal':'sf','up':'sfup','down':'sfdown'}[variation],wp,eta[mask],pt[mask])
            if not np.isfinite(values).all() or np.any(values<=0): raise ValueError('Invalid lepton SF')
            result[component+'_'+variation] = ak.to_numpy(ak.prod(ak.unflatten(values,counts),axis=1))
    return result


def veto_flags(events,diphoton,electrons,muons,jets,cfg):
    flags = {}
    if cfg['tau_veto'] != 'off':
        from .wwgg_tau_veto import evaluate
        tc = json.loads((ROOT / cfg['tau_config']).read_text())
        tc['mode'] = cfg['tau_veto']
        data = evaluate(events.Tau,diphoton,electrons,muons,tc)
        if bool(ak.any(data['tau_trial_n_invalid']>0)):
            raise ValueError('Invalid Tau input/DeepTau encoding; do not silently accept veto')
        for k,v in data.items():
            if v.ndim==1: flags[k.replace('tau_trial_','tau_')] = v
    if cfg['b_veto'] != 'off':
        from .wwgg_b_veto import evaluate_b,load_b_config
        data = evaluate_b(jets,load_b_config(ROOT))
        if bool(ak.any(data['b_trial_n_invalid_kin']>0)):
            raise ValueError('Invalid UParT score in b-veto phase space')
        for k,v in data.items():
            if v.ndim==1: flags[k.replace('b_trial_','b_')] = v
    false = ak.zeros_like(events.event,dtype=bool)
    flags.setdefault('tau_veto',false)
    flags.setdefault('b_veto',false)
    return flags
