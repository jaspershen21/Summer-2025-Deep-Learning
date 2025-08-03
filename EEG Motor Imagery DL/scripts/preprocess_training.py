import mne
from mne.preprocessing import ICA
import os
import scipy.io

RAW_DATA_DIR = "./../data/raw/"
PROCESSED_DATA_DIR = "./../data/processed/"
SAVED_ICA_DIR = "./../data/ica/"
TRUE_LABELS_DIR = "./../data/true labels/"

#
# TRAINING DATA PREPROCESSING PIPELINE
#

# Get file names for training data
raw_training_file_names = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.endswith(".gdf") and "T" in f])

for raw_file_name in raw_training_file_names:
    print(f"Processing {raw_file_name}...")

    SUBJECT_ID = raw_file_name.split("T")[0]
    ICA_FILE_PATH = os.path.join(SAVED_ICA_DIR, f"{SUBJECT_ID}-ica.fif")
    PROCESSED_FILE_PATH = os.path.join(PROCESSED_DATA_DIR, f"{SUBJECT_ID}T-epo.fif")
    TRUE_LABEL_PATH = os.path.join(TRUE_LABELS_DIR, f"{SUBJECT_ID}T.mat")

    # Load file
    path_to_file = os.path.join(RAW_DATA_DIR, raw_file_name)
    raw = mne.io.read_raw_gdf(path_to_file, preload = True, eog = ["EOG-left", "EOG-central", "EOG-right"])

    # Band Pass Filter (0.5 - 40 Hz)
    # 50Hz Notch Filter already applied during data collection
    raw.filter(l_freq = 0.5, h_freq = 40.0)

    # Independent Component Analysis (ICA)
    n_EEG_channels = sum(1 for ch in raw.info["ch_names"] if ch.startswith("EEG"))
    ica = ICA(n_components = n_EEG_channels, random_state = 42)
    ica.fit(raw)

    eog_indices, eog_scores = ica.find_bads_eog(raw)
    ica.save(ICA_FILE_PATH, overwrite = True)
    ica.apply(raw, exclude = eog_indices)

    # Epoching data by events
    events, event_id = mne.events_from_annotations(raw)

    epochs = mne.Epochs(
        raw,
        events,
        event_id = [event_id["769"], event_id["770"], event_id["771"], event_id["772"]],
        tmin = -0.5,
        tmax = 3.5,
        picks = "eeg",
        reject_by_annotation = True,
        preload = True,
        baseline = (None, 0)
    )

    true_labels = scipy.io.loadmat(TRUE_LABEL_PATH)["classlabel"].flatten()
    epochs.events[:, 2] = true_labels

    epochs.save(PROCESSED_FILE_PATH, overwrite = True)

print("Preprocessing of training data complete.")