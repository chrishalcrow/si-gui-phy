import json
from spikeinterface_gui import run_mainwindow
import shutil
from pathlib import Path
import numpy as np

from probeinterface import read_prb

from spikeinterface.extractors import read_phy
from spikeinterface.core import SortingAnalyzer, generate_ground_truth_recording, create_sorting_analyzer, ChannelSparsity
import spikeinterface.qualitymetrics as siqm

extensions_to_compute = {
    "unit_locations": {"method": "center_of_mass"},
    "correlograms": {},
    "template_similarity": {},
    "isi_histograms": {},
    "template_metrics": {}
}

def make_amplitudes(sa: SortingAnalyzer, phy_path: Path):

    amps_folder = Path(sa.folder) / 'extensions' / 'spike_amplitudes'
    amps_folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(phy_path/"amplitudes.npy", amps_folder / "amplitudes.npy")

    make_run_info(amps_folder)

    params = {
        "peak_sign": "pos"
    }
    make_params(amps_folder, params)

def make_locations(sa, phy_path):

    locs_folder: Path = sa.folder / 'extensions' / 'spike_locations'
    locs_folder.mkdir(parents=True, exist_ok=True)

    locs_np = np.load(phy_path / "spike_positions.npy")

    num_dims = len(locs_np[0])

    column_names = ['x', 'y', 'z'][:num_dims]
    dtype = [(name, locs_np.dtype) for name in column_names]
    structured_array = np.array([tuple(row) for row in locs_np], dtype=dtype)

    # Save as new .npy file
    np.save(locs_folder / "spike_locations.npy", structured_array)

    make_run_info(locs_folder)

    params = {}
    make_params(locs_folder, params)

def make_sparsity(sort, rec, phy_path):

    templates = np.load(phy_path / "templates.npy")
    
    unit_ids = sort.unit_ids
    channel_ids = rec.channel_ids

    mask = np.sum(templates, axis=1) != 0
    sparsity = ChannelSparsity(mask, unit_ids=unit_ids, channel_ids=channel_ids)
    return sparsity

def make_templates(sa, phy_path, unwhiten=True):

    templates_folder = Path(sa.folder) / Path('extensions/templates')
    templates_folder.mkdir(parents=True, exist_ok=True)

    whitened_templates = np.load(phy_path / "templates.npy")
    wh_inv = np.load(phy_path / "whitening_mat_inv.npy")

    if unwhiten:
        new_templates = compute_unwhitened_templates(whitened_templates, wh_inv)
    else:
        new_templates = whitened_templates

    np.save(templates_folder / "average.npy", new_templates)

    make_run_info(templates_folder)

    params = {
        'operators': ['average'],
        'ms_before': 1.0,
        'ms_after': 1.06,
        'peak_sign': "pos",
    }
    make_params(templates_folder, params)


from numba import njit

@njit()
def compute_unwhitened_templates(whitened_templates, wh_inv):
 
    template_shape = np.shape(whitened_templates)
    new_templates = np.zeros(template_shape) 
    for a in range(template_shape[0]):
        for b in range(template_shape[2]):
            for c in range(template_shape[2]):
                new_templates[a,:,b] += wh_inv[b,c]*whitened_templates[a,:,c]
    
    return new_templates

def make_run_info(folder):

    run_info = {
        "run_completed": True,
        "runtime_s": 0,
    }
    with open(folder / 'run_info.json', 'w') as f:
        json.dump(run_info, f)

def make_params(folder, params):

    with open(folder / 'params.json', 'w') as f:
        json.dump(params, f)


def analyzer_from_phy(phy_path, verbose=True) -> SortingAnalyzer:

    print("Verified that you have a phy folder made with KiloSort4!")
    print("Making a **recordingless** SortingAnalyzer")
    print("Reading the probe information...")
    
    probe = read_prb(phy_path/ "probe.prb")

    print("Reading the sorting...")

    sort = read_phy(phy_path)
    duration = sort.to_spike_vector()[-1][0]/sort.sampling_frequency
    recording, _ = generate_ground_truth_recording(probe=probe.probes[0], sampling_frequency=30_000, durations=[duration])

    sparsity = make_sparsity(sort, recording, phy_path)

    print("Creating a temporary folder...")
    sa = create_sorting_analyzer(sort,recording, format="binary_folder", folder="temp_analyzer", overwrite=True, sparse=True, sparsity=sparsity)

    print("Copying and reformatting templates", end="")
    sa.compute("random_spikes")
    make_templates(sa, phy_path, unwhiten=True)
    print(", locations", end="")
    make_locations(sa, phy_path)
    print(" and ampltides...")
    make_amplitudes(sa, phy_path)

    print("Computing unit locations, correlograms, template similarity, isi histograms and some template and quality metrics.")
    sa.compute(extensions_to_compute)
    siqm.compute_firing_rates(sa)
    siqm.compute_presence_ratios(sa)
    siqm.compute_isi_violations(sa)
    siqm.compute_refrac_period_violations(sa)
    siqm.compute_sliding_rp_violations(sa)
    siqm.compute_amplitude_medians(sa)
    siqm.compute_amplitude_cv_metrics(sa)
    siqm.compute_synchrony_metrics(sa)
    siqm.compute_firing_ranges(sa)
    siqm.compute_drift_metrics(sa)

    #sa._recording = None

    return sa


   
