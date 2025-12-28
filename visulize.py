import matplotlib.pyplot as plt
import numpy as np


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
