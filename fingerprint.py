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


def generate_hashes(peaks: np.ndarray, fan_value: int = 5,
                    time_window: int = 50) -> list[int]:
    """
    Generate hashes from peak coordinates.

    Args:
        peaks: A 2D array of (frequency, time) pairs.
        fan_value: Number of peaks to consider.
        time_window: Maximum time distance between peaks.

    Returns:
        A list of hashes.
    """
    hashes = []
    num_peaks = len(peaks)

    for i in range(num_peaks):
        freq1, t1 = peaks[i]

        for j in range(i + 1, min(i + 1 + fan_value, num_peaks)):
            freq2, t2 = peaks[j]

            if t2 - t1 > time_window:
                break

            delta_t = t2 - t1
            hash_value = hash((freq1, freq2, delta_t))

            hashes.append(hash_value)

    return hashes
