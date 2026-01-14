import matplotlib.pyplot as plt
import numpy as np
import mne


class DataLoader:
    def __init__(self, file_path, inspect=False):
        self.file_path = file_path
        raw = mne.io.read_raw_edf(self.file_path, preload=True)
        # print("raw: ", raw.info)
        self.data = raw.get_data()
        self.channel_names = raw.ch_names
        self.sampling_rate = raw.info['sfreq']
        self.time = np.arange(len(self.data)) / self.sampling_rate
        self.channel_names = raw.ch_names
        self.n_channels = len(self.channel_names)
        self.epochs = None
        self.events = None
        self.y = None
        if inspect:          
            raw.plot(duration=60, n_channels=self.n_channels, proj=False, scalings='auto', remove_dc=True)
            print("n_channels:", self.n_channels)
            print("channel names:", self.channel_names)
            print("sampling rate:", self.sampling_rate)
            print("time shape:", self.time)
            print("time shape 2:", raw.times)
            print("data shape:", self.data.shape)
            print("RAW ANOTAION: ", raw.annotations)
            print("without montage", raw.info['chs'][0])
            self.inspect_annotations(raw.annotations)
        self.preprocess(self.data) # or raw ?
        self.set_events_epochs()
        self.features = self.compute_frequency_domain()


    def inspect_annotations(self, annotations):
        print("Annotations:", annotations)
        for annot in annotations:
            print(f"  Onset: {annot['onset']}, Duration: {annot['duration']:.2f}s, Description: {annot['description']}")


    def preprocess(self, raw):
        """
        Rename channels to standard 10-20 system.
        Apply band-pass filter (8-40 Hz) and notch filter.
        """
        raw.rename_channels({ch: ch.strip('.').upper().replace('Z', 'z').replace('FP', 'Fp') for ch in raw.ch_names})
        raw.set_montage('standard_1020')
        fig = raw.plot_sensors(show_names=True, show=False)
        plt.savefig("images/montage_plt.png")
        plt.close(fig)
        print("with montage ", raw.info['chs'][0])

        self.filtered = raw.copy()
        self.filtered.filter(l_freq=8, h_freq=40)
        self.filtered.notch_filter(freqs=60)
        self.filtered.set_eeg_reference('average')
        print("Filtered data shape:", self.filtered.get_data().shape)


    def set_events_epochs(self):
        """
        Find events and extract epochs.
        """
        events, event_id = mne.events_from_annotations(raw)
        print("Event IDs:", event_id)
        print("First 5 events:", events[:5])
        epochs = mne.Epochs(
            self.filtered,
            events,
            event_id=event_id,
            baseline=None,
            preload=True
        )
        self.epochs = epochs
        print("Epochs info:", epochs.info)
        self.y = epochs.events[:, 2]
        print("Epochs data shape:", epochs.get_data().shape)


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
            print("PSD shape:", psd.shape)
            print("PSD:", psd)

        return np.array(features)


def save_features_labels(self):
    """
    Save extracted features and labels to .npy files.
    """
    x = self.features
    print("Extracted features shape:", x.shape)
    y = self.y
    np.save("data/X_train.npy", x)
    np.save("data/y_train.npy", y)


def main():
    # model = DataLoader("/home/gkrusta/tpv/S002R04.edf")
    # model = DataLoader("/home/gkrusta/physionet.org/files/eegmmidb/1.0.0/S005/S005R07.edf")
    # model = DataLoader("/sgoinfre/students/gkrusta/tpv/S002R04.edf")
    model = DataLoader("/sgoinfre/students/gkrusta/physionet.org/files/eegmmidb/1.0.0/S006/S006R06.edf")


if __name__ == "__main__":
    main()
