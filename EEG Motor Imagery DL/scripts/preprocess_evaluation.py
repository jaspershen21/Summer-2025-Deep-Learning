import mne
from mne.preprocessing import ICA
import os
import scipy.io

RAW_DATA_DIR = "./../data/raw/"
PROCESSED_DATA_DIR = "./../data/processed/"
SAVED_ICA_DIR = "./../data/ica/"
TRUE_LABELS_DIR = "./../data/true labels/"

#
# EVALUATION DATA PREPROCESSING PIPELINE
#

# Get file names for evaluation data
raw_evaluation_file_names = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".gdf") and "E" in f])

for raw_file_name in raw_evaluation_file_names:
    print(f"Processing {raw_file_name}...")

    SUBJECT_ID = raw_file_name.split("E")[0]
    ICA_FILE_PATH = os.path.join(SAVED_ICA_DIR, f"{SUBJECT_ID}-ica.fif")
    PROCESSED_FILE_PATH = os.path.join(PROCESSED_DATA_DIR, f"{SUBJECT_ID}E-epo.fif")
    TRUE_LABEL_PATH = os.path.join(TRUE_LABELS_DIR, f"{SUBJECT_ID}E.mat")

    # Load file
    path_to_file = os.path.join(RAW_DATA_DIR, raw_file_name)
    raw = mne.io.read_raw_gdf(path_to_file, preload = True, eog = ["EOG-left", "EOG-central", "EOG-right"])

    # Band Pass Filter (0.5 - 40 Hz)
    # 50Hz Notch Filter already applied during data collection
    raw.filter(l_freq = 0.5, h_freq = 40.0)

    # Independent Component Analysis (ICA)
    ica = mne.preprocessing.read_ica(ICA_FILE_PATH)
    ica.apply(raw)

    # Epoching data by events
    events, _ = mne.events_from_annotations(raw, event_id = {"783": 783})

    # Label epochs with true labels
    true_labels = scipy.io.loadmat(TRUE_LABEL_PATH)["classlabel"].flatten()
    label_map = {1: 769, 2: 770, 3: 771, 4: 772}
    mapped_labels = [label_map[label] for label in true_labels]
    events[:, 2] = mapped_labels
    
    epochs = mne.Epochs(
        raw,
        events,
        event_id = {"left": 769, "right": 770, "foot": 771, "tongue": 772},
        tmin = -0.5,
        tmax = 3.5,
        preload = True,
        baseline = (None, 0)
    )

    epochs.save(PROCESSED_FILE_PATH, overwrite = True)