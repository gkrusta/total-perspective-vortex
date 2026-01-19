from pyexpat import model
import mne
import numpy as np


EXPERIMENTS = {
    # 'both_hands_vs_feet'
    0: {
        'runs': [5, 9, 13],
        'events': {'T1': 2, 'T2': 3}
    },
    # 'imagine_both_hands_vs_feet'
    1: {
        'runs': [6, 10, 14],
        'events': {'T1': 2, 'T2': 3}
    },
    # 'left_hand_vs_right_hand'
    2: {
        'runs': [3, 7, 11],
        'events': {'T1': 2, 'T2': 3}
    },
    # 'imagine_left_hand_vs_right_hand'
    3: {
        'runs': [4, 8, 12],
        'events': {'T1': 2, 'T2': 3}
    },
    # 'rest_vs_both_hands'
    4: {
        'runs': [5, 9, 13],
        'events': {'T0': 1, 'T1': 2}
    },
    # 'rest_vs_imagine_both_hands'
    5: {
        'runs': [6, 10, 14],
        'events': {'T0': 1, 'T1': 2}
    },
}

USEFULL_CHANNELS = ['Cz', 'FCz', 'CPz', 'C3', 'C4', 'FC3', 'FC4', 'CP3', 'CP4']

# BAD_SUBJECTS = []


def open_subject(subject, run):
    """
    Format subject and run IDs to have leading zeros.
    """
    try:
        subject_id = str(subject).zfill(3)
        run_id = str(run).zfill(2)
        file_path = f"/sgoinfre/students/gkrusta/physionet.org/files/eegmmidb/1.0.0/S{subject_id}/S{subject_id}R{run_id}.edf"
        # file_path = f"/home/gkrusta/physionet.org/files/eegmmidb/1.0.0/S{subject_id}/S{subject_id}R{run_id}.edf"

        print("Opening subject:", file_path)
        raw = mne.io.read_raw_edf(file_path, preload=True)
        data = raw.get_data()
        print("raw: ", raw.info)
        print("data: ",  data.shape)
        return raw
    except Exception as e:
        print(f"Error opening subject {subject}, run {run}: {e}")
        exit(1)


def describe_data(model):
    print("Data shape:", model.data.shape)
    print("Data stats: min =", np.min(model.data), ", max =", np.max(model.data), ", mean =", np.mean(model.data), ", std =", np.std(model.data))

    # model.raw.plot(duration=60, n_channels=model.n_channels, proj=False, scalings='auto', remove_dc=True)
    
    print("n_channels:", model.n_channels)
    print("channel names:", model.channel_names)
    print("sampling rate:", model.sampling_rate)
    print("time marks:", model.time)
    print("data shape:", model.data.shape)

    print("Annotations:", model.raw.annotations)
    for annot in model.raw.annotations:
        print(f"  Onset: {annot['onset']}, Duration: {annot['duration']:.2f}s, Description: {annot['description']}")
    
    print("Features:", model.features.shape)
    print("Labels:", model.labels.shape)
    print("Epochs data shape:", model.epochs.get_data().shape)
    print("Epochs info:", model.epochs.info)


def find_events(model,experiment_dict):
    """
    Returns a mask to filter events for t1, t2 or t3 according to
    experiment.
    """
    events = experiment_dict['events']
    mask = (model.labels == list(events.values())[0]) | (model.labels == list(events.values())[1])
    return mask
