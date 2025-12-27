import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.decomposition import PCA
import mne

class DataLoader:
    def __init__(self, file_path):
        self.file_path = file_path
        raw = mne.io.read_raw_edf(self.file_path, preload=True)
        if self.file_path.endswith('.event'):
            events, event_id = mne.events_from_annotations(self.file_path)
        # print("raw: ", raw.info)
        self.data = raw.get_data()
        self.channel_names = raw.ch_names
        self.sampling_rate = raw.info['sfreq']
        self.time = np.arange(len(self.data)) / self.sampling_rate
        self.channel_names = raw.ch_names
        self.n_channels = len(self.channel_names)
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
        print("Epochs data shape:", epochs.get_data().shape)
        fig = epochs.plot(n_epochs=5, show=True)
        plt.show()
        plt.savefig("images/epochs_plt.png")
        plt.close(fig)
        print("EPOCHS", epochs)


    def visualize(self, file):
        plt.plot(self.time, self.data)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.title('EEG Signal - Channel 0')
        # plt.show()


def main():
    # model = PCA(n_components=5)
    # x_new = model.fit_transform(x)
    model = DataLoader("/home/gkrusta/physionet.org/files/eegmmidb/1.0.0/S002/S002R04.edf")
    # model.visualize(model.file_path)


if __name__ == "__main__":
    main()
