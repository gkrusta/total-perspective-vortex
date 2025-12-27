import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.decomposition import PCA
import mne
# ensure images directory exists
os.makedirs("images", exist_ok=True)

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
        fig.set_size_inches(18, 18)
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

    def compute_spectrum(self, channel: int = 0, n_fft: int | None = None, plot: bool = True, save_path: str | None = None, freq_limit: float | None = 60.0):
        """Compute a simple FFT-based power spectrum for a single channel.

        - channel: channel index to analyse
        - n_fft: length of FFT (defaults to full signal length)
        - plot: whether to save a plot of the PSD
        - save_path: explicit path to save plot (defaults to images/fft_channel_{channel}.png)
        - freq_limit: x-axis limit in Hz for plotting (use None to show full range)
        Returns: (freqs, psd)
        """
        # choose source: filtered data if available, otherwise raw
        source = self.filtered
        if source is not None:
            data = source.get_data()
        else:
            data = self.data

        if channel < 0 or channel >= data.shape[0]:
            raise IndexError(f"channel index {channel} out of range (0..{data.shape[0]-1})")

        sig = data[channel]
        N = len(sig)
        if n_fft is None:
            n_fft = N

        # apply a Hann window to reduce spectral leakage
        window = np.hanning(N)
        sig_win = sig * window

        fft = np.fft.rfft(sig_win, n=n_fft)
        psd = (np.abs(fft) ** 2) / np.sum(window ** 2)
        freqs = np.fft.rfftfreq(n_fft, d=1.0 / self.sampling_rate)

        if plot:
            fig = plt.figure(figsize=(10, 6))
            plt.semilogy(freqs, psd, lw=1)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power')
            plt.title(f'PSD (FFT) - Channel {channel}')
            if freq_limit is not None:
                plt.xlim(0, min(freq_limit, self.sampling_rate / 2))
            out = save_path or f"images/fft_channel_{channel}.png"
            plt.tight_layout()
            plt.savefig(out)
            plt.close(fig)

        return freqs, psd


def main():
    # model = PCA(n_components=5)
    # x_new = model.fit_transform(x)
    # model = DataLoader("/home/gkrusta/physionet.org/files/eegmmidb/1.0.0/S002/S002R04.edf")
    model = DataLoader("/sgoinfre/students/gkrusta/tpv/S002R04.edf")
    # demo: compute FFT on channel 0 and save PSD plot
    try:
        model.compute_spectrum(channel=0, save_path="images/fft_channel_0.png", freq_limit=60.0)
        print("Saved FFT PSD to images/fft_channel_0.png")
    except Exception as e:
        print("Failed to compute or save FFT:", e)
    # model.visualize(model.file_path)


if __name__ == "__main__":
    main()
