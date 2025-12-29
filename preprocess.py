import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.decomposition import PCA
import mne
# ensure images directory exists
os.makedirs("images", exist_ok=True)

bands = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta":  (13, 30),
}


class DataLoader:
    def __init__(self, file_path):
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
        # raw.plot(duration=60, n_channels=self.n_channels, proj=False, scalings='auto', remove_dc=True)
        # print("n_channels:", self.n_channels)
        # print("channel names:", self.channel_names)
        # print("sampling rate:", self.sampling_rate)
        # print("time shape:", self.time)
        # print("time shape 2:", raw.times)
        # print("data shape:", self.data.shape)
        # print("RAW ANOTAION: ", raw.annotations)
        # print("without montage", raw.info['chs'][0])
        self.preprocess(raw)
        self.inspect_annotations(raw.annotations)


    def inspect_annotations(self, annotations):
        print("Annotations:", annotations)
        for annot in annotations:
            print(f"  Onset: {annot['onset']}, Duration: {annot['duration']:.2f}s, Description: {annot['description']}")


    def preprocess(self, raw):
        # rename channels to standard 10-20 system
        raw.rename_channels({ch: ch.strip('.').upper().replace('Z', 'z').replace('FP', 'Fp') for ch in raw.ch_names})
        raw.set_montage('standard_1020')
        fig = raw.plot_sensors(show_names=True, show=False)
        plt.savefig("images/montage_plt.png")
        plt.close(fig)
        # print("with montage ", raw.info['chs'][0])

        # apply band pass filter
        self.filtered = raw.copy()
        self.filtered.filter(l_freq=1, h_freq=40)
        self.filtered.notch_filter(freqs=60)
        self.filtered.set_eeg_reference('average')
        # print("Filtered data shape:", self.filtered.get_data().shape)
        fig = self.filtered.plot(n_channels=10, duration=8, scalings='auto', show=False)
        plt.savefig("images/filtered_reference_plt.png")
        plt.close(fig)
        fig = raw.plot(n_channels=10, duration=8, scalings='auto', show=False)
        plt.savefig("images/raw_plt.png")
        plt.close(fig)

        # find events and extract epochs
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
        self.y = epochs.events[:, 2]
        print("Epochs data shape:", epochs.get_data().shape)
        fig = epochs.plot(n_epochs=5, show=True)
        fig.set_size_inches(18, 18)
        plt.savefig("images/epochs_plt.png")
        plt.show()
        plt.close(fig)


    def visualize(self, file):
        plt.plot(self.time, self.data)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('EEG Signal - Channel 0')
        # plt.show()


    def mark_bandpower(self):
        data = self.epochs.get_data()
        n_epochs, n_channels, n_times = data.shape
        features = []

        for epoch in data:
            psd, freqs = mne.time_frequency.psd_array_welch(
                epoch,
                sfreq=self.sampling_rate,
                fmin=1,
                fmax=40,
                n_fft=n_times,
                window='hann',
            )

            epoch_features = []
            for fmin, fmax in bands.values():
                idx = np.logical_and(freqs >= fmin, freqs <= fmax)
                band_power = psd[:, idx].mean(axis=1)
                epoch_features.extend(band_power)

            features.append(epoch_features)

        return np.array(features)


def main():
    # model = PCA(n_components=5)
    # x_new = model.fit_transform(x)
    # model = DataLoader("/home/gkrusta/tpv/S002R04.edf")
    # model = DataLoader("/home/gkrusta/physionet.org/files/eegmmidb/1.0.0/S005/S005R07.edf")
    # model = DataLoader("/sgoinfre/students/gkrusta/tpv/S002R04.edf")
    model = DataLoader("/sgoinfre/students/gkrusta/physionet.org/files/eegmmidb/1.0.0/S006/S006R06.edf")

    # demo: compute FFT and PSD 
    x = model.mark_bandpower()
    y = model.y
    np.save("data/X_train.npy", x)
    np.save("data/y_train.npy", y)
    # model.visualize(model.file_path)


if __name__ == "__main__":
    main()
