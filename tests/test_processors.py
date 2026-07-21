import subprocess
import json
import pytest
from importlib import resources
from higgs_dna.workflows import HggBaseProcessor, HggFiducialProcessor, TagAndProbeProcessor, HHbbggProcessor, HplusCharmProcessor, LowMassProcessor, ParticleLevelProcessor, TopProcessor, ZeeProcessor, ZmmyProcessor, STXSProcessor, BTaggingEfficienciesProcessor
from coffea import processor
from coffea.nanoevents import NanoAODSchema


# Not tested by default since does not start with "test_"
def run_processor(processor_instance, fileset):
    """
    Helper function to run a given processor instance on the provided fileset.
    """
    iterative_run = processor.Runner(
        executor=processor.IterativeExecutor(compression=None),
        schema=NanoAODSchema,
    )
    out = iterative_run(
        fileset=fileset,
        treename="Events",
        processor_instance=processor_instance,
    )
    return out


@pytest.mark.parametrize("processor_class", [
    HggBaseProcessor,
    HggFiducialProcessor,
    TagAndProbeProcessor,
    HHbbggProcessor,
    # Hpc Cannot be included in a simple way here since the arguments are not defaulted
    #HplusCharmProcessor,
    # Unclear to me why low mass does not work here, unit test should also be designed for this processor
    #LowMassProcessor,
    ParticleLevelProcessor,
    TopProcessor,
    ZeeProcessor,
    #ZmmyProcessor,
    BTaggingEfficienciesProcessor,
    STXSProcessor,
])
def test_processors(processor_class):
    """
    Test that each processor can run over a basic nanoAOD data v11 file without errors.
    """
    # Pull the golden JSON file
    subprocess.run("pull_files.py --target GoldenJSON", shell=True)

    # Pull the jetID files
    subprocess.run("pull_files.py --target JetMET", shell=True)

    # Pull the btagging SF files
    subprocess.run("pull_files.py --target bTag", shell=True)

    # Need to pull some JSONs and ONNXs for HHbbggProcessor
    subprocess.run("xrdcp root://eoscms.cern.ch//eos/cms/store/group/phys_b2g/HHbbgg/evourlio/WPs_btagging.json higgs_dna/tools/WPs_btagging_HHbbgg.json", shell=True)
    subprocess.run("xrdcp root://eoscms.cern.ch//eos/cms/store/group/phys_b2g/HHbbgg/nkasarag/HiggsDNA_JSONs/Weights_interference.json higgs_dna/tools/Weights_interference_HHbbgg.json", shell=True)
    subprocess.run("xrdcp root://eoscms.cern.ch//eos/cms/store/group/phys_b2g/HHbbgg/jafan/mbbModels/mjj_model_2023.onnx higgs_dna/tools/mjj_model_2023.onnx", shell=True)
    subprocess.run("xrdcp root://eoscms.cern.ch//eos/cms/store/group/phys_b2g/HHbbgg/jafan/mbbModels/mjj_model_2022.onnx higgs_dna/tools/mjj_model_2022.onnx", shell=True)
    subprocess.run("xrdcp root://eoscms.cern.ch//eos/cms/store/group/phys_b2g/HHbbgg/lindo/HHbbgg_bpairing_Run2_allyears.onnx higgs_dna/tools/HHbbgg_bpairing_Run2_allyears.onnx", shell=True)
    subprocess.run("xrdcp root://eoscms.cern.ch//eos/cms/store/group/phys_b2g/HHbbgg/lindo/HHbbgg_bpairing_Run3_allyears.onnx higgs_dna/tools/HHbbgg_bpairing_Run3_allyears.onnx", shell=True)
    subprocess.run("xrdcp root://eoscms.cern.ch//eos/cms/store/group/phys_b2g/HHbbgg/chouy/vbf_pairing/HHbbgg_vbfpairing_Run3.onnx higgs_dna/tools/HHbbgg_vbfpairing_Run3.onnx", shell=True)

    # Choose datasets to run over appropriately
    # In the future, should specify datasets on eos instead of local files
    # These should be appropriate for the processor being tested (e.g. muon for Zmmy or DY for T&P)
    # The skeleton below should be adjusted

    if processor_class == HggBaseProcessor or processor_class == HggFiducialProcessor:
        MC = "./tests/samples/skimmed_nano/ggH_M125_amcatnlo_v13.root"
        Data = "./tests/samples/skimmed_nano/EGamma_2022E_v13.root"
    elif processor_class == TagAndProbeProcessor:
        MC = None
        Data = "./tests/samples/skimmed_nano/EGamma_2022E_v13.root"
    elif processor_class == HHbbggProcessor:
        MC = "./tests/samples/skimmed_nano/ggH_M125_amcatnlo_v13.root"
        Data = "./tests/samples/skimmed_nano/EGamma_2022E_v13.root"
    elif processor_class == HplusCharmProcessor:
        MC = "./tests/samples/skimmed_nano/ggH_M125_amcatnlo_v13.root"
        Data = "./tests/samples/skimmed_nano/EGamma_2022E_v13.root"
    elif processor_class == LowMassProcessor:
        MC = "./tests/samples/skimmed_nano/ggH_M125_amcatnlo_v13.root"
        Data = "./tests/samples/skimmed_nano/EGamma_2022E_v13.root"
    elif processor_class == ParticleLevelProcessor:
        MC = "./tests/samples/skimmed_nano/ggH_M125_amcatnlo_v13.root"
        Data = None
    elif processor_class == TopProcessor:
        MC = "./tests/samples/skimmed_nano/ggH_M125_amcatnlo_v13.root"
        Data = "./tests/samples/skimmed_nano/EGamma_2022E_v13.root"
    elif processor_class == ZeeProcessor:
        MC = "./tests/samples/skimmed_nano/ggH_M125_amcatnlo_v13.root"
        Data = "./tests/samples/skimmed_nano/EGamma_2022E_v13.root"
    elif processor_class == ZmmyProcessor:
        MC = None
        Data = None
    elif processor_class == BTaggingEfficienciesProcessor:
        MC = "./tests/samples/skimmed_nano/ggH_M125_amcatnlo_v13.root"
        Data = None
    elif processor_class == STXSProcessor:
        MC = "./tests/samples/skimmed_nano/ggH_M125_amcatnlo_v13.root"
        Data = "./tests/samples/skimmed_nano/EGamma_2022E_v13.root"

    fileset = {}

    if Data is not None:
        fileset["Data"] = [Data]

    if MC is not None:
        fileset["MC"] = [MC]

    with resources.open_text("higgs_dna.metaconditions", "Era2017_legacy_v1.json") as f:
        metaconditions = json.load(f)

    processor_instance = processor_class(
        year={"Data": ["2022postEE"], "MC": ["2022postEE"]},
        metaconditions=metaconditions,
        nano_version=13,
        apply_trigger=True,
        skipJetVetoMap=True,
        output_location="output/basics"
    )

    # Run the processor and verify output
    run_processor(processor_instance, fileset)

    # Clean up
    subprocess.run("rm -r output", shell=True)
