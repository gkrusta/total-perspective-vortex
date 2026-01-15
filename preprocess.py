import numpy as np
import mne
from utils import open_subject, describe_data

class DataLoader:
    def __init__(self, subject, run, inspect=False):
        raw = open_subject(subject, run)
        self.raw = raw
        self.data = raw.get_data()
        self.channel_names = raw.ch_names
        self.sampling_rate = raw.info['sfreq']
        self.time = np.arange(len(self.data)) / self.sampling_rate
        self.channel_names = raw.ch_names
        self.n_channels = len(self.channel_names)
        self.epochs = None
        self.events = None
        self.labels = None
        self.preprocess(raw)
        self.set_events_epochs(raw)
        self.features = self.compute_frequency_domain()
        if inspect:          
            describe_data(self)


    def preprocess(self, raw):
        """
        Rename channels to standard 10-20 system.
        Apply band-pass filter (8-40 Hz) and notch filter.
        """
        raw.rename_channels({ch: ch.strip('.').upper().replace('Z', 'z').replace('FP', 'Fp') for ch in raw.ch_names})
        raw.set_montage('standard_1020')

        self.filtered = raw.copy()
        self.filtered.filter(l_freq=8, h_freq=40)
        self.filtered.notch_filter(freqs=60)
        self.filtered.set_eeg_reference('average')


    def set_events_epochs(self, raw):
        """
        Find events and extract epochs.
        """
        events, event_id = mne.events_from_annotations(raw)
        epochs = mne.Epochs(
            self.filtered,
            events,
            tmin=-0.1,
            tmax=4.0,
            event_id=event_id,
            baseline=None,
            preload=True
        )
        self.epochs = epochs
        self.labels = epochs.events[:, 2]


    def compute_frequency_domain(self):
        """
        Converts time-domain epochs to frequency-domain features using Welch's method.
        It tells how much power (on average) each frequency has in that channel, over the entire signal used.
        Returns: features array of shape (n_epochs, n_channels, n_freqs)
        """
        data = self.epochs.get_data()
        n_epochs, n_channels, n_times = data.shape
        features = []
        for epoch in data:
            psd, freqs = mne.time_frequency.psd_array_welch(
                epoch,
                sfreq=self.sampling_rate,
                fmin=8,
                fmax=40,
                n_fft=n_times,
                window='hann',
            )
            features.append(psd)

        return np.array(features)


# def save_features_labels(self):
#     """
#     Save extracted features and labels to .npy files.
#     """
#     x = self.features
#     y = self.y
#     print("Extracted features shape:", x.shape, y.shape)
#     np.save("data/X_train.npy", x)
#     np.save("data/y_train.npy", y)


def main():
    # model = DataLoader("/home/gkrusta/tpv/S002R04.edf")
    # model = DataLoader("/home/gkrusta/physionet.org/files/eegmmidb/1.0.0/S005/S005R07.edf")
    # model = DataLoader("/sgoinfre/students/gkrusta/tpv/S002R04.edf")
    model = DataLoader("/sgoinfre/students/gkrusta/physionet.org/files/eegmmidb/1.0.0/S006/S006R06.edf")


if __name__ == "__main__":
    main()
