## Instructions on how to use the MuonPOG corrections

Under each data taking campaign, there will be 3 files with corrections: `muon_JPsi.json`, `muon_Z.json`, `muon_HighPt.json`, 
depending on the method used to derive them and the pT range that they cover, as explained in the [MuonPOG Wiki](https://muon-wiki.docs.cern.ch/) for each year.
A fourth file, called `muon_scalesmearing.json` is also present (for Run3 campaigns) and it contains the scale and resolution corrections to be applied on the muon pT.
See the table below for more details on those files.

| Correction file | Method used to derive SFs |      pT range     | Wiki link |
|:---------------:|:-------------------------:|:-----------------:|:----------:|
| `muon_JPsi`     | TnP on J/Psi peak         | pT < 30 GeV       | [low-pt](https://muon-wiki.docs.cern.ch/guidelines/corrections/#low-pt-pt-30-gev)    |
| `muon_Z`        | TnP on Z peak             | 15 < pT < 200 GeV | [medium-pt](https://muon-wiki.docs.cern.ch/guidelines/corrections/#medium-pt-30-gev-pt-200-gev) |
| `muon_HighPt`   | CutnCount on high-mass DY | pT > 200 GeV      | [high-pt](https://muon-wiki.docs.cern.ch/guidelines/corrections/#high-pt-pt-200-gev)   |
| `muon_scalesmearing` | Z peak matching described [here](https://cds.cern.ch/record/2904701/files/DP2024_065.pdf) | pT < 200 GeV     | [corrections](https://muon-wiki.docs.cern.ch/code/ptcorr/) |

**Important Note:** Since Run 3 2023, SFs at the Z peak are computed as a function of eta, with more granularity, instead of the usual abs(eta). For all the previous years (Run 2 UL + 2022), and for all the pT regimes, even though SFs are computed as a function of abs(eta), it is possible to read them using eta as input instead, for consistency with 2023. Please, refer to the `muonExample.py` for more details about the usage.

### Scale and smearing corrections

The scale and smearing corrections (`muon_scalesmearing.json`) can be applied as shown in the examples provided under `examples/muoScaleAndSmearingRDFExample.py` or `examples/muoScaleAndSmearingCoffeaExample.py`, depending on whether you use `RDataFrame` or `coffea`.

**Important Note:** These corrections are not provided as SFs, but are computed on the fly depending on the pT, eta, phi, and number of layers in the tracker of each muon in the event. The computation is performed by the functions defined in `examples/MuonScaRe.*` (where both `.cc` and `.py` versions are available) which must be imported in your analysis code as shown in the two examples provided.
