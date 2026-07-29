import sys
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from MuonScaRe import pt_resol, pt_scale, pt_resol_var, pt_scale_var 
import correctionlib
# Only needed to dump root output file
import uproot

if len(sys.argv) != 2:
    print("Usage: python muoScaleAndSmearingCoffeaExample.py <correction_file.json>")
    sys.exit(1)

correction_file = sys.argv[1]

events_data = NanoEventsFactory.from_root(
    'root://cms-xrd-global.cern.ch//store/data/Run2022C/Muon/NANOAOD/22Sep2023-v1/2520000/393889f7-173f-48ba-ad46-94828a4ee4b2.root:Events',
    schemaclass=NanoAODSchema,
).events()

events_mc = NanoEventsFactory.from_root("root://cms-xrd-global.cern.ch//store/mc/Run3Summer22NanoAODv12/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5-v2/40000/4fa63a35-456c-4d69-9400-356033e505f1.root:Events", schemaclass=NanoAODSchema).events()

cset = correctionlib.CorrectionSet.from_file(f"{correction_file}")

# Data: only scale correction to gen Z peak
events_data["Muon", "ptcorr"] = pt_scale(
    1, # 1 for data, 0 for mc 
    events_data.Muon.pt, 
    events_data.Muon.eta, 
    events_data.Muon.phi, 
    events_data.Muon.charge, 
    cset, 
    nested=True # for awkward arrays. Set False for 1d arrays
)

# MC: both scale correction to gen Z peak AND resolution correction to Z width in data
events_mc["Muon", "ptscalecorr"] = pt_scale(
    0, 
    events_mc.Muon.pt, 
    events_mc.Muon.eta, 
    events_mc.Muon.phi, 
    events_mc.Muon.charge, 
    cset, 
    nested=True
)

events_mc["Muon", "ptcorr"] = pt_resol(
    events_mc.Muon.ptscalecorr, 
    events_mc.Muon.eta, 
    events_mc.Muon.phi,
    events_mc.Muon.nTrackerLayers, 
    events_mc.event,
    events_mc.luminosityBlock,
    cset,
    nested=True
)

# uncertainties
events_mc["Muon", "ptscalecorr_up"] = pt_scale_var(
    events_mc.Muon.ptcorr, 
    events_mc.Muon.eta, 
    events_mc.Muon.phi, 
    events_mc.Muon.charge,
    "up",
    cset, 
    nested=True
)
events_mc["Muon", "ptscalecorr_dn"] = pt_scale_var(
    events_mc.Muon.ptcorr, 
    events_mc.Muon.eta, 
    events_mc.Muon.phi, 
    events_mc.Muon.charge,
    "dn",
    cset, 
    nested=True
)

events_mc["Muon", "ptcorr_resolup"] = pt_resol_var(
    events_mc.Muon.ptscalecorr, 
    events_mc.Muon.ptcorr, 
    events_mc.Muon.eta, 
    "up",
    cset, 
    nested=True
)
events_mc["Muon", "ptcorr_resoldn"] = pt_resol_var(
    events_mc.Muon.ptscalecorr, 
    events_mc.Muon.ptcorr, 
    events_mc.Muon.eta, 
    "dn",
    cset, 
    nested=True
)


# Store output in TTree for debug/inspection
branches_to_save = {
    "Muon_pt": events_mc.Muon.pt,
    "Muon_eta": events_mc.Muon.eta,
    "Muon_phi": events_mc.Muon.phi,
    "event": events_mc.event,
    "luminosityBlock": events_mc.luminosityBlock,
    "Muon_ptcorr": events_mc.Muon.ptcorr,
    "Muon_ptcorr_resolup": events_mc.Muon.ptcorr_resolup,
    "Muon_ptcorr_resoldn": events_mc.Muon.ptcorr_resoldn,
}

with uproot.recreate("coffea_corrections.root") as f:
    f["Events"] = branches_to_save

