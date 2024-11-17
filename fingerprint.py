import numpy as np
from skimage.feature import peak_local_max
from skimage import img_as_float, data


def find_local_peaks(
        spectrogram: np.ndarray, min_distance: int = 20) -> np.ndarray:
    """
    Find local peaks in a spectrogram.

    Args:
        spectrogram: The spectrogram.
        min_distance: Minimum distance between peaks.

    Returns:
        An array of peak coordinates.
    """
    return peak_local_max(spectrogram, min_distance=min_distance)


if __name__ == "__main__":
    from spectrogram import generate_spectrogram
    import matplotlib.pyplot as plt

    S_db, sr = generate_spectrogram()

    im = img_as_float(S_db)
    peaks = find_local_peaks(im)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3), sharex=True, sharey=True)
    ax = axes.ravel()

    ax[0].imshow(im, cmap=plt.cm.gray, origin='lower', aspect='auto')
    ax[0].axis('off')
    ax[0].set_title('Original')

    ax[1].imshow(im, cmap=plt.cm.gray, origin='lower', aspect='auto')
    ax[1].autoscale(False)
    ax[1].plot(peaks[:, 1], peaks[:, 0], 'r.')
    ax[1].axis('off')
    ax[1].set_title('Peak local max')

    fig.tight_layout()

    plt.show()
