import numpy as np
import librosa


def generate_spectrogram(
        audio_path: str | None = None) -> tuple[np.ndarray, int]:
    """
    Generate a spectrogram from an audio file.

    Args:
        audio_path: Path to the audio file.

    Returns:
        A 2-tuple containing the spectrogram and the sample rate.
    """
    y, sr = librosa.load(audio_path if audio_path
                         else librosa.ex("trumpet"), sr=None)
    D = librosa.stft(y)
    S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

    return S_db, sr
