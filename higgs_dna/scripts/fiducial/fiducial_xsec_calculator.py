### Examples of command line
# python fiducial_xsec_calculator.py PATH_TO_FOLDER_WITH_ParticleLevelProcessor_OUTPUT --process all --bin '|0|0.15|0.3|0.6|0.9|2.5|' --era all --obs YH
## Without NNLOPS reweighing
# python fiducial_xsec_calculator.py /work/atarabin/HiggsDNA/prod/particleLevel/ --process ggH --bin '|0|1|2|3|100|' --era all --obs NJ --weight "genWeight"
## POWHEG
# python fiducial_xsec_calculator.py /work/atarabin/HiggsDNA/prod/particleLevel/ --process ggH --bin '|0|15|30|45|80|120|200|350|1000|' --era all --powheg
## For inclusive XS
# python fiducial_xsec_calculator.py /work/atarabin/HiggsDNA/prod/particleLevel/ --process all --bin '|0|500|' --era all
import argparse
import awkward as ak
import numpy as np
from scipy import interpolate


def safe_divide(num, den):
    """Return num/den, guarding against division by zero."""
    return 0.0 if den == 0 else num / den


def evaluate_value_at_mass(values, mass_points, no_interpolation, target_mass=125.38):
    """
    Evaluate a quantity defined at discrete mass points at a desired target mass.
    If interpolation is disabled, the 125 GeV value is returned.
    """
    if not values:
        return 0.0
    if no_interpolation:
        key = "125" if "125" in values else mass_points[0]
        return values[key]
    mass_value_pairs = sorted((float(m), values[m]) for m in mass_points)
    masses, points = zip(*mass_value_pairs)
    order = min(2, len(masses) - 1)
    if order < 1:
        return points[0]
    spline = interpolate.splrep(masses, points, k=order)
    return float(interpolate.splev(target_mass, spline))


def compute_fid_xsec(in_frac, mass_points, xs_value, BR, no_interpolation, target_mass=125.38):
    """
    Compute the fiducial cross section for a given observable.

    Parameters:
      in_frac (dict): Dictionary of in-fiducial fractions keyed by mass point.
      mass_points (list): List of mass points (as strings) provided via the command line.
      xs_value (float): Cross section value from the XS map for the process.
      BR (float): Branching ratio.
      no_interpolation (bool): Flag to disable interpolation when only mass point 125 is provided.
      target_mass (float): The mass at which to evaluate the spline (default 125.38).
    
    Returns:
      float: The computed fiducial cross section.
    """
    value = evaluate_value_at_mass(in_frac, mass_points, no_interpolation, target_mass=target_mass)
    return value * xs_value * 1000 * BR



available_processes = ['ggH', 'VBFH', 'VH', 'ttH', 'all', 'xH']
available_mass_points = ['120', '125', '130']
available_fid_selections = ['fiducialGeometricFlag', 'fiducialClassicalFlag']
available_years = ['2022']
available_eras = ['preEE', 'postEE', 'all']

# Setup command-line argument parsing
parser = argparse.ArgumentParser(description = "Calculate the inclusive fiducial cross section of pp->H(yy)+X process(es) based on processed samples without detector-level selections. ")
parser.add_argument('path', type = str, help = "Path to the top-level folder containing the different directories. Please only run this on the output of the ParticleLevelProcessor.")
parser.add_argument('--process', type = str, choices = available_processes, default = 'ggH', help = "Please specify the process(es) for which you want to calculate the inclusive fiducial xsec.")
parser.add_argument('--mass-points', nargs='+', choices = available_mass_points, default = available_mass_points, help = "Please specify the mass points to run over. If only one single mass point of 125 is specified: No interpolation is performed.")
parser.add_argument('--fid-selection', type = str, choices = available_fid_selections, default = 'fiducialGeometricFlag', help = "Please specify the fiducial selection flag to use.")
parser.add_argument('--year', type = str, choices = available_years, default = '2022', help = 'Please specify the desired year if you want to combine samples from multiple eras.')
parser.add_argument('--era', type = str, choices = available_eras, default = 'postEE', help = "Please specify the era(s) that you want to run over. If you specify 'all', an inverse variance weighting is performed to increase the precision.")
parser.add_argument('--bin', type = str, default = '|0|5000|', help = "Bin boundaries of the differential XS. The default")
parser.add_argument('--obs', type = str, default = 'PTH', help = "Name of the differential observable: PTH, YH, NJ")
parser.add_argument('--weight', type = str, default = 'weight', help = "Weight to use.")
parser.add_argument('--powheg', action="store_true", help="To process powheg sample.")
parser.add_argument('--per-process-output', action="store_true", help="Also store the fiducial cross sections and acceptances for each process separately.")

args = parser.parse_args()

args = parser.parse_args()

# New check for mass points
if len(args.mass_points) == 1:
    if args.mass_points[0] != '125':
        parser.error("When specifying a single mass point, only '125' is allowed.")
    no_interpolation = True
else:
    no_interpolation = False

if args.fid_selection == 'fiducialGeometricFlag':
    print('INFO: Using the geometric fiducial flag for the selection.')
elif args.fid_selection == 'fiducialClassicalFlag':
    print('INFO: Using the classical fiducial flag for the selection.')

path_folder = args.path # Use the specified folder path
# Pepare the processes array appropriately
if args.process == 'all':
    processes = [p for p in available_processes if p not in ('all', 'xH')]
elif args.process == 'xH':
    processes = ['VBFH', 'VH', 'ttH']
else:
    processes = [args.process]
year = args.year
# Pepare the eras array appropriately
if args.era == 'all':
    eras = [e for e in available_eras if e != 'all']
else:
    eras = [args.era]

# Convert bin boundaries in a list
obs_bins = [float(num) for num in args.bin.strip("|").split("|")]

# See also the following pages (note numbers always in picobarn)
# 13: for 125
# 13p6: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/LHCHWG136TeVxsec_extrap, for 125.38
# 14: for 125
XS_map = {'13':   {'ggH': 48.58, 'VBFH': 3.782, 'VH': 2.2569, 'ttH': 0.5071},
         '13p6': {'ggH': 51.96, 'VBFH': 4.067, 'VH': 2.3781, 'ttH': 0.5638},
         '14':   {'ggH': 54.67, 'VBFH': 4.278, 'VH': 2.4991, 'ttH': 0.6137},}

XS_map_scale_up = {'13':   {},
                   '13p6': {'ggH': 53.98644, 'VBFH': 4.087335, 'VH': 2.3376723, 'ttH': 0.597628},
                   '14':   {},}

XS_map_scale_dn = {'13':   {},
                   '13p6': {'ggH': 49.93356, 'VBFH': 4.054799, 'VH': 2.4185277, 'ttH': 0.5113666},
                   '14':   {},}

XS_map_pdf_up = {'13':   {},
                   '13p6': {'ggH': 52.94724, 'VBFH': 4.152407, 'VH': 2.4137715, 'ttH': 0.580714},
                   '14':   {},}

XS_map_pdf_dn = {'13':   {},
                   '13p6': {'ggH': 50.97276, 'VBFH': 3.981593, 'VH': 2.3424285, 'ttH': 0.546886},
                   '14':   {},}

XS_map_alphaS_up = {'13':   {},
                   '13p6': {'ggH': 53.31096, 'VBFH': 4.087335, 'VH': 2.3995029, 'ttH': 0.575076},
                   '14':   {},}

XS_map_alphaS_dn = {'13':   {},
                    '13p6': {'ggH': 50.60904, 'VBFH': 4.046665, 'VH': 2.3566971, 'ttH': 0.552524},
                    '14':   {},}


def combine_acceptances(process_acceptances):
    """Cross-section weighted combination using nominal 13.6 TeV cross sections."""
    total_sigma = 0.0
    weighted_sum = 0.0
    for process, acc in process_acceptances.items():
        sigma = XS_map['13p6'].get(process, 0.0)
        total_sigma += sigma
        weighted_sum += sigma * acc
    return safe_divide(weighted_sum, total_sigma)


# This depends on how you named your samples in HiggsDNA
processMap = {'ggH':  'GluGluHtoGG',
              'VBFH': 'VBFHtoGG',
              'VH':   'VHtoGG',
              'ttH':   'ttHtoGG',}

BR = 0.2270/100 # SM value for mH close to 125: https://twiki.cern.ch/twiki/bin/view/LHCPhysics/CERNYellowReportPageBR
mass_points = args.mass_points # The fiducial acceptance should be extrapolated at 125.38 using a spline between 120, 125, and 130 if we have the mass points
# For powheg only the 125 GeV sample is available
# [FIXME] For the time being, no extrapolation, its effects is in any case small.
# [FIXME] Idea: compute the relative variation between 125 and 125.38 GeV with Madgraph and then apply it to powheg
mass_powheg = {120:125, 125:125, 130:125}
scale_weight_indices = [0, 1, 3, 5, 7, 8]
pdf_weight_indices = list(range(1, 101))
alpha_weight_map = {'alpha_up': 101, 'alpha_dn': 102}

fid_xsecs_per_bin = {}
fid_xsecs_per_bin_scale_up = {}
fid_xsecs_per_bin_scale_dn = {}
fid_xsecs_per_bin_pdf_up = {}
fid_xsecs_per_bin_pdf_dn = {}
fid_xsecs_per_bin_alpha_up = {}
fid_xsecs_per_bin_alpha_dn = {}
acc_per_bin = {}
acc_per_bin_scale_up = {}
acc_per_bin_scale_dn = {}
acc_per_bin_pdf_up = {}
acc_per_bin_pdf_dn = {}
acc_per_bin_alpha_up = {}
acc_per_bin_alpha_dn = {}
per_process_fid_xsecs = {}
per_process_acc = {}
if args.per_process_output:
    fid_keys = ['fidXS', 'fidXS_scale_up', 'fidXS_scale_dn', 'fidXS_pdf_up', 'fidXS_pdf_dn', 'fidXS_alpha_up', 'fidXS_alpha_dn']
    acc_keys = ['Acc', 'Acc_scale_up', 'Acc_scale_dn', 'Acc_pdf_up', 'Acc_pdf_dn', 'Acc_alpha_up', 'Acc_alpha_dn']
    per_process_fid_xsecs = {key: {process: [] for process in processes} for key in fid_keys}
    per_process_acc = {key: {process: [] for process in processes} for key in acc_keys}
for b in range(len(obs_bins)-1):
    fid_xsecs_per_bin_process = {}
    fid_xsecs_per_bin_process_scale_up = {}
    fid_xsecs_per_bin_process_scale_dn = {}
    fid_xsecs_per_bin_process_pdf_up = {}
    fid_xsecs_per_bin_process_pdf_dn = {}
    fid_xsecs_per_bin_process_alpha_up = {}
    fid_xsecs_per_bin_process_alpha_dn = {}
    acc_per_bin_process = {}
    acc_per_bin_process_scale_up = {}
    acc_per_bin_process_scale_dn = {}
    acc_per_bin_process_pdf_up = {}
    acc_per_bin_process_pdf_dn = {}
    acc_per_bin_process_alpha_up = {}
    acc_per_bin_process_alpha_dn = {}
    for process in processes:
        print(f'INFO: Now extracting fraction of in-fiducial events for process {process} ...')
        in_frac_per_mass = {}
        in_frac_per_mass_scale_up = {}
        in_frac_per_mass_scale_dn = {}
        in_frac_per_mass_pdf_up = {}
        in_frac_per_mass_pdf_dn = {}
        in_frac_per_mass_alpha_up = {}
        in_frac_per_mass_alpha_dn = {}
        for mass in mass_points:
            print(f'INFO: Now extracting numbers for mass {mass}...')
            sumw_nom_in = 0.0
            sumw_nom_all = 0.0
            scale_sums = {idx: {'num': 0.0, 'den': 0.0} for idx in scale_weight_indices}
            pdf_sums = {idx: {'num': 0.0, 'den': 0.0} for idx in pdf_weight_indices}
            alpha_sums = {key: {'num': 0.0, 'den': 0.0} for key in alpha_weight_map}
            for era in eras:
                print(f'INFO: Now extracting numbers for era {era}, process {process}, and bin {b} ...')
                # Extract the events
                process_mass = str(mass)
                process_string = path_folder + processMap[process] + '_M-' + process_mass + '_' + era
                if args.powheg:
                    process_mass_key = int(mass)
                    process_string = path_folder + processMap[process] + '_M-' + str(mass_powheg[process_mass_key]) + '_powheg'
                arr = ak.from_parquet(process_string)
                # Calculating the relevant fractions
                inFiducialFlag = (arr[args.fid_selection] == True) & (abs(arr[args.obs]) >= obs_bins[b]) & (abs(arr[args.obs]) < obs_bins[b+1]) # Only for this type of tagger right now, can be customised in the future

                weights = arr[args.weight]
                sumw_nom_all += float(ak.sum(weights))
                sumw_nom_in += float(ak.sum(weights[inFiducialFlag]))

                for idx in scale_weight_indices:
                    scale_weight = arr["LHEScaleWeight_"+str(idx)]
                    scale_sums[idx]['den'] += float(ak.sum(weights * scale_weight))
                    scale_sums[idx]['num'] += float(ak.sum(weights[inFiducialFlag] * scale_weight[inFiducialFlag]))

                for idx in pdf_weight_indices:
                    pdf_weight = arr["LHEPdfWeight_"+str(idx)]
                    pdf_sums[idx]['den'] += float(ak.sum(weights * pdf_weight))
                    pdf_sums[idx]['num'] += float(ak.sum(weights[inFiducialFlag] * pdf_weight[inFiducialFlag]))

                for alpha_key, weight_idx in alpha_weight_map.items():
                    alpha_weight = arr["LHEPdfWeight_"+str(weight_idx)]
                    alpha_sums[alpha_key]['den'] += float(ak.sum(weights * alpha_weight))
                    alpha_sums[alpha_key]['num'] += float(ak.sum(weights[inFiducialFlag] * alpha_weight[inFiducialFlag]))

            in_frac = safe_divide(sumw_nom_in, sumw_nom_all)

            scale_acceptances = [safe_divide(scale_sums[idx]['num'], scale_sums[idx]['den']) for idx in scale_weight_indices]
            in_frac_scale_up = max(scale_acceptances) if scale_acceptances else in_frac
            in_frac_scale_dn = min(scale_acceptances) if scale_acceptances else in_frac

            pdf_acceptances = np.array([safe_divide(pdf_sums[idx]['num'], pdf_sums[idx]['den']) for idx in pdf_weight_indices])
            ## Formula taken from https://arxiv.org/pdf/1510.03865 (Eq. 20)
            pdf_uncertainty = np.sqrt(np.sum(np.square(pdf_acceptances - in_frac)))
            in_frac_pdf_up = in_frac + pdf_uncertainty
            in_frac_pdf_dn = in_frac - pdf_uncertainty

            in_frac_alpha_up = safe_divide(alpha_sums['alpha_up']['num'], alpha_sums['alpha_up']['den'])
            in_frac_alpha_dn = safe_divide(alpha_sums['alpha_dn']['num'], alpha_sums['alpha_dn']['den'])

            print(f"INFO: Combined fraction of in-fiducial events: {in_frac} ...")
            print(f"INFO: Combined fraction scale up: {in_frac_scale_up} ...")
            print(f"INFO: Combined fraction scale dn: {in_frac_scale_dn} ...")
            print(f"INFO: Combined fraction pdf up: {in_frac_pdf_up} ...")
            print(f"INFO: Combined fraction pdf dn: {in_frac_pdf_dn} ...")
            print(f"INFO: Combined fraction alpha up: {in_frac_alpha_up} ...")
            print(f"INFO: Combined fraction alpha dn: {in_frac_alpha_dn} ...")

            in_frac_per_mass[mass] = in_frac
            in_frac_per_mass_scale_up[mass] = in_frac_scale_up
            in_frac_per_mass_scale_dn[mass] = in_frac_scale_dn
            in_frac_per_mass_pdf_up[mass] = in_frac_pdf_up
            in_frac_per_mass_pdf_dn[mass] = in_frac_pdf_dn
            in_frac_per_mass_alpha_up[mass] = in_frac_alpha_up
            in_frac_per_mass_alpha_dn[mass] = in_frac_alpha_dn

        acc_per_bin_process[process] = evaluate_value_at_mass(in_frac_per_mass, args.mass_points, no_interpolation)
        acc_per_bin_process_scale_up[process] = evaluate_value_at_mass(in_frac_per_mass_scale_up, args.mass_points, no_interpolation)
        acc_per_bin_process_scale_dn[process] = evaluate_value_at_mass(in_frac_per_mass_scale_dn, args.mass_points, no_interpolation)
        acc_per_bin_process_pdf_up[process] = evaluate_value_at_mass(in_frac_per_mass_pdf_up, args.mass_points, no_interpolation)
        acc_per_bin_process_pdf_dn[process] = evaluate_value_at_mass(in_frac_per_mass_pdf_dn, args.mass_points, no_interpolation)
        acc_per_bin_process_alpha_up[process] = evaluate_value_at_mass(in_frac_per_mass_alpha_up, args.mass_points, no_interpolation)
        acc_per_bin_process_alpha_dn[process] = evaluate_value_at_mass(in_frac_per_mass_alpha_dn, args.mass_points, no_interpolation)

        fid_xsecs_per_bin_process[process] = acc_per_bin_process[process] * XS_map['13p6'][process] * 1000 * BR
        fid_xsecs_per_bin_process_scale_up[process] = acc_per_bin_process_scale_up[process] * XS_map_scale_up['13p6'][process] * 1000 * BR
        fid_xsecs_per_bin_process_scale_dn[process] = acc_per_bin_process_scale_dn[process] * XS_map_scale_dn['13p6'][process] * 1000 * BR
        fid_xsecs_per_bin_process_pdf_up[process] = acc_per_bin_process_pdf_up[process] * XS_map_pdf_up['13p6'][process] * 1000 * BR
        fid_xsecs_per_bin_process_pdf_dn[process] = acc_per_bin_process_pdf_dn[process] * XS_map_pdf_dn['13p6'][process] * 1000 * BR
        fid_xsecs_per_bin_process_alpha_up[process] = acc_per_bin_process_alpha_up[process] * XS_map_alphaS_up['13p6'][process] * 1000 * BR
        fid_xsecs_per_bin_process_alpha_dn[process] = acc_per_bin_process_alpha_dn[process] * XS_map_alphaS_dn['13p6'][process] * 1000 * BR

    if args.per_process_output:
        for process in processes:
            per_process_fid_xsecs['fidXS'][process].append(fid_xsecs_per_bin_process[process])
            per_process_fid_xsecs['fidXS_scale_up'][process].append(fid_xsecs_per_bin_process_scale_up[process])
            per_process_fid_xsecs['fidXS_scale_dn'][process].append(fid_xsecs_per_bin_process_scale_dn[process])
            per_process_fid_xsecs['fidXS_pdf_up'][process].append(fid_xsecs_per_bin_process_pdf_up[process])
            per_process_fid_xsecs['fidXS_pdf_dn'][process].append(fid_xsecs_per_bin_process_pdf_dn[process])
            per_process_fid_xsecs['fidXS_alpha_up'][process].append(fid_xsecs_per_bin_process_alpha_up[process])
            per_process_fid_xsecs['fidXS_alpha_dn'][process].append(fid_xsecs_per_bin_process_alpha_dn[process])

            per_process_acc['Acc'][process].append(acc_per_bin_process[process])
            per_process_acc['Acc_scale_up'][process].append(acc_per_bin_process_scale_up[process])
            per_process_acc['Acc_scale_dn'][process].append(acc_per_bin_process_scale_dn[process])
            per_process_acc['Acc_pdf_up'][process].append(acc_per_bin_process_pdf_up[process])
            per_process_acc['Acc_pdf_dn'][process].append(acc_per_bin_process_pdf_dn[process])
            per_process_acc['Acc_alpha_up'][process].append(acc_per_bin_process_alpha_up[process])
            per_process_acc['Acc_alpha_dn'][process].append(acc_per_bin_process_alpha_dn[process])

    fid_xsecs_per_bin[b] = np.sum(np.asarray([fid_xsecs_per_bin_process[process] for process in processes]))
    print(f"The fiducial cross section for {args.obs} in [{obs_bins[b]},{obs_bins[b+1]}] is: {fid_xsecs_per_bin[b]} fb")

    fid_xsecs_per_bin_scale_up[b] = np.sum(np.asarray([fid_xsecs_per_bin_process_scale_up[process] for process in processes]))
    print(f"The fiducial cross section (scale_up) for {args.obs} in [{obs_bins[b]},{obs_bins[b+1]}] is: {fid_xsecs_per_bin_scale_up[b]} fb")

    fid_xsecs_per_bin_scale_dn[b] = np.sum(np.asarray([fid_xsecs_per_bin_process_scale_dn[process] for process in processes]))
    print(f"The fiducial cross section (scale_dn) for {args.obs} in [{obs_bins[b]},{obs_bins[b+1]}] is: {fid_xsecs_per_bin_scale_dn[b]} fb")

    fid_xsecs_per_bin_pdf_up[b] = np.sum(np.asarray([fid_xsecs_per_bin_process_pdf_up[process] for process in processes]))
    print(f"The fiducial cross section (pdf_up) for {args.obs} in [{obs_bins[b]},{obs_bins[b+1]}] is: {fid_xsecs_per_bin_pdf_up[b]} fb")

    fid_xsecs_per_bin_pdf_dn[b] = np.sum(np.asarray([fid_xsecs_per_bin_process_pdf_dn[process] for process in processes]))
    print(f"The fiducial cross section (pdf_dn) for {args.obs} in [{obs_bins[b]},{obs_bins[b+1]}] is: {fid_xsecs_per_bin_pdf_dn[b]} fb")

    fid_xsecs_per_bin_alpha_up[b] = np.sum(np.asarray([fid_xsecs_per_bin_process_alpha_up[process] for process in processes]))
    print(f"The fiducial cross section (alpha_up) for {args.obs} in [{obs_bins[b]},{obs_bins[b+1]}] is: {fid_xsecs_per_bin_alpha_up[b]} fb")

    fid_xsecs_per_bin_alpha_dn[b] = np.sum(np.asarray([fid_xsecs_per_bin_process_alpha_dn[process] for process in processes]))
    print(f"The fiducial cross section (alpha_dn) for {args.obs} in [{obs_bins[b]},{obs_bins[b+1]}] is: {fid_xsecs_per_bin_alpha_dn[b]} fb")

    acc_per_bin[b] = combine_acceptances(acc_per_bin_process)
    print(f"The acceptance for {args.obs} in [{obs_bins[b]},{obs_bins[b+1]}] at 125.38 GeV is: {acc_per_bin[b]}")

    acc_per_bin_scale_up[b] = combine_acceptances(acc_per_bin_process_scale_up)
    print(f"The acceptance (scale_up) for {args.obs} in [{obs_bins[b]},{obs_bins[b+1]}] at 125.38 GeV is: {acc_per_bin_scale_up[b]}")

    acc_per_bin_scale_dn[b] = combine_acceptances(acc_per_bin_process_scale_dn)
    print(f"The acceptance (scale_dn) for {args.obs} in [{obs_bins[b]},{obs_bins[b+1]}] at 125.38 GeV is: {acc_per_bin_scale_dn[b]}")

    acc_per_bin_pdf_up[b] = combine_acceptances(acc_per_bin_process_pdf_up)
    print(f"The acceptance (pdf_up) for {args.obs} in [{obs_bins[b]},{obs_bins[b+1]}] at 125.38 GeV is: {acc_per_bin_pdf_up[b]}")

    acc_per_bin_pdf_dn[b] = combine_acceptances(acc_per_bin_process_pdf_dn)
    print(f"The acceptance (pdf_dn) for {args.obs} in [{obs_bins[b]},{obs_bins[b+1]}] at 125.38 GeV is: {acc_per_bin_pdf_dn[b]}")

    acc_per_bin_alpha_up[b] = combine_acceptances(acc_per_bin_process_alpha_up)
    print(f"The acceptance (alpha_up) for {args.obs} in [{obs_bins[b]},{obs_bins[b+1]}] at 125.38 GeV is: {acc_per_bin_alpha_up[b]}")

    acc_per_bin_alpha_dn[b] = combine_acceptances(acc_per_bin_process_alpha_dn)
    print(f"The acceptance (alpha_dn) for {args.obs} in [{obs_bins[b]},{obs_bins[b+1]}] at 125.38 GeV is: {acc_per_bin_alpha_dn[b]}")


final_fid_xsec = np.sum(np.asarray([fid_xsecs_per_bin[b] for b in range(len(obs_bins)-1)]))
print(f"The inclusive fiducial cross section is given by: {final_fid_xsec} fb")

final_fid_xsec_scale_up = np.sum(np.asarray([fid_xsecs_per_bin_scale_up[b] for b in range(len(obs_bins)-1)]))
print(f"The inclusive fiducial cross section (scale up) is given by: {final_fid_xsec_scale_up} fb")

final_fid_xsec_scale_dn = np.sum(np.asarray([fid_xsecs_per_bin_scale_dn[b] for b in range(len(obs_bins)-1)]))
print(f"The inclusive fiducial cross section (scale dn) is given by: {final_fid_xsec_scale_dn} fb")

final_fid_xsec_pdf_up = np.sum(np.asarray([fid_xsecs_per_bin_pdf_up[b] for b in range(len(obs_bins)-1)]))
print(f"The inclusive fiducial cross section (pdf_up) is given by: {final_fid_xsec_pdf_up} fb")

final_fid_xsec_pdf_dn = np.sum(np.asarray([fid_xsecs_per_bin_pdf_dn[b] for b in range(len(obs_bins)-1)]))
print(f"The inclusive fiducial cross section (pdf_dn) is given by: {final_fid_xsec_pdf_dn} fb")

final_fid_xsec_alpha_up = np.sum(np.asarray([fid_xsecs_per_bin_alpha_up[b] for b in range(len(obs_bins)-1)]))
print(f"The inclusive fiducial cross section (alpha_up) is given by: {final_fid_xsec_alpha_up} fb")

final_fid_xsec_alpha_dn = np.sum(np.asarray([fid_xsecs_per_bin_alpha_dn[b] for b in range(len(obs_bins)-1)]))
print(f"The inclusive fiducial cross section (alpha_dn) is given by: {final_fid_xsec_alpha_dn} fb")

output = 'fidXS_'+args.obs+'_'+args.process
if args.powheg: output += '_powheg'
if args.weight != "weight": output += '_'+args.weight

with open(output+'.py', 'w') as f:
        f.write('Boundaries = '+str(obs_bins)+' \n')
        f.write('fidXS = '+str(list(fid_xsecs_per_bin.values()))+' \n')
        f.write('fidXS_scale_up = '+str(list(fid_xsecs_per_bin_scale_up.values()))+' \n')
        f.write('fidXS_scale_dn = '+str(list(fid_xsecs_per_bin_scale_dn.values()))+' \n')
        f.write('fidXS_pdf_up = '+str(list(fid_xsecs_per_bin_pdf_up.values()))+' \n')
        f.write('fidXS_pdf_dn = '+str(list(fid_xsecs_per_bin_pdf_dn.values()))+' \n')
        f.write('fidXS_alpha_up = '+str(list(fid_xsecs_per_bin_alpha_up.values()))+' \n')
        f.write('fidXS_alpha_dn = '+str(list(fid_xsecs_per_bin_alpha_dn.values()))+' \n')
        f.write('Acc = '+str(list(acc_per_bin.values()))+' \n')
        f.write('Acc_scale_up = '+str(list(acc_per_bin_scale_up.values()))+' \n')
        f.write('Acc_scale_dn = '+str(list(acc_per_bin_scale_dn.values()))+' \n')
        f.write('Acc_pdf_up = '+str(list(acc_per_bin_pdf_up.values()))+' \n')
        f.write('Acc_pdf_dn = '+str(list(acc_per_bin_pdf_dn.values()))+' \n')
        f.write('Acc_alpha_up = '+str(list(acc_per_bin_alpha_up.values()))+' \n')
        f.write('Acc_alpha_dn = '+str(list(acc_per_bin_alpha_dn.values()))+' \n')
        if args.per_process_output:
            for process in processes:
                f.write(f'fidXS_{process} = {per_process_fid_xsecs["fidXS"][process]} \n')
                f.write(f'fidXS_scale_up_{process} = {per_process_fid_xsecs["fidXS_scale_up"][process]} \n')
                f.write(f'fidXS_scale_dn_{process} = {per_process_fid_xsecs["fidXS_scale_dn"][process]} \n')
                f.write(f'fidXS_pdf_up_{process} = {per_process_fid_xsecs["fidXS_pdf_up"][process]} \n')
                f.write(f'fidXS_pdf_dn_{process} = {per_process_fid_xsecs["fidXS_pdf_dn"][process]} \n')
                f.write(f'fidXS_alpha_up_{process} = {per_process_fid_xsecs["fidXS_alpha_up"][process]} \n')
                f.write(f'fidXS_alpha_dn_{process} = {per_process_fid_xsecs["fidXS_alpha_dn"][process]} \n')
                f.write(f'Acc_{process} = {per_process_acc["Acc"][process]} \n')
                f.write(f'Acc_scale_up_{process} = {per_process_acc["Acc_scale_up"][process]} \n')
                f.write(f'Acc_scale_dn_{process} = {per_process_acc["Acc_scale_dn"][process]} \n')
                f.write(f'Acc_pdf_up_{process} = {per_process_acc["Acc_pdf_up"][process]} \n')
                f.write(f'Acc_pdf_dn_{process} = {per_process_acc["Acc_pdf_dn"][process]} \n')
                f.write(f'Acc_alpha_up_{process} = {per_process_acc["Acc_alpha_up"][process]} \n')
                f.write(f'Acc_alpha_dn_{process} = {per_process_acc["Acc_alpha_dn"][process]} \n')
