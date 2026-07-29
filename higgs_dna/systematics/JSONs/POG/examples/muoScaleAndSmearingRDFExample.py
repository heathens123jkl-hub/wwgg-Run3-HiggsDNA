import sys
import ROOT
import correctionlib

correctionlib.register_pyroot_binding()

ROOT.EnableImplicitMT()

if len(sys.argv) != 2:
    print("Usage: python muoScaleAndSmearingRDFExample.py <correction_file.json>")
    sys.exit(1)

correction_file = sys.argv[1]

ROOT.gROOT.ProcessLine(
    f'auto cset = correction::CorrectionSet::from_file("{correction_file}");'
)
ROOT.gROOT.ProcessLine('#include "MuonScaRe.cc"')

df_data = ROOT.RDataFrame("Events", "root://cms-xrd-global.cern.ch//store/data/Run2022C/Muon/NANOAOD/22Sep2023-v1/2520000/393889f7-173f-48ba-ad46-94828a4ee4b2.root")
df_mc = ROOT.RDataFrame("Events", "root://cms-xrd-global.cern.ch//store/mc/Run3Summer22NanoAODv12/DYto2L-2Jets_MLL-50_TuneCP5_13p6TeV_amcatnloFXFX-pythia8/NANOAODSIM/130X_mcRun3_2022_realistic_v5-v2/40000/4fa63a35-456c-4d69-9400-356033e505f1.root")

df_data = df_data.Define("pt_1", "Muon_pt.size() > 0 ? Muon_pt[0] : -999.")
df_data = df_data.Define("eta_1", "Muon_eta.size() > 0 ? Muon_eta[0] : -999.")
df_data = df_data.Define("phi_1", "Muon_phi.size() > 0 ? Muon_phi[0] : -999.")
df_data = df_data.Define("charge_1", "Muon_charge.size() > 0 ? Muon_charge[0] : 0")

df_mc = df_mc.Define("pt_1", "Muon_pt.size() > 0 ? Muon_pt[0] : -999.")
df_mc = df_mc.Define("eta_1", "Muon_eta.size() > 0 ? Muon_eta[0] : -999.")
df_mc = df_mc.Define("phi_1", "Muon_phi.size() > 0 ? Muon_phi[0] : -999.")
df_mc = df_mc.Define("charge_1", "Muon_charge.size() > 0 ? Muon_charge[0] : 0")
df_mc = df_mc.Define("nTrkLayers_1", "Muon_nTrackerLayers.size() > 0 ? Muon_nTrackerLayers[0] : -999")

# Data apply scale correction
df_data = df_data.Define(
    'pt_1_corr',
    'pt_scale(1, pt_1, eta_1, phi_1, charge_1)'
)

# MC apply scale shift and resolution smearing
df_mc = df_mc.Define(
    'pt_1_scale_corr',
    'pt_scale(0, pt_1, eta_1, phi_1, charge_1)'
)
df_mc = df_mc.Define(
    'pt_1_corr',
    'pt_resol(pt_1_scale_corr, eta_1, phi_1, float(nTrkLayers_1), event, luminosityBlock)'
)

# MC evaluate scale uncertainty
df_mc = df_mc.Define(
    'pt_1_scale_corr_up',
    'pt_scale_var(pt_1_corr, eta_1, phi_1, charge_1, "up")'
)
df_mc = df_mc.Define(
    'pt_1_scale_corr_dn',
    'pt_scale_var(pt_1_corr, eta_1, phi_1, charge_1, "dn")'
)

# MC evaluate resolution uncertainty
df_mc = df_mc.Define(
    "pt_1_corr_resolup",
    'pt_resol_var(pt_1_scale_corr, pt_1_corr, eta_1, "up")'
)
df_mc = df_mc.Define(
    "pt_1_corr_resoldn",
    'pt_resol_var(pt_1_scale_corr, pt_1_corr, eta_1, "dn")'
)

# Dump some variables to root file for inspection/debug
output_file = "rdf_corrections.root"
df_mc.Snapshot(
    "Events",
    output_file,
    ["pt_1", "eta_1", "phi_1", "event", "luminosityBlock", "pt_1_corr"]
)
