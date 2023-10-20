import numpy as np

from libsoni.util.utils import normalize_signal, fade_signal, smooth_weights
from libsoni.core.methods import generate_sinusoid


def sonify_spectrogram(spectrogram: np.ndarray,
                       frequency_coefficients: np.ndarray = None,
                       time_coefficients: np.ndarray = None,
                       fading_duration: float = 0.05,
                       sonification_duration: int = None,
                       normalize: bool = True,
                       fs: int = 22050) -> np.ndarray:
    """Sonifies a spectrogram using sinusoids.

    Parameters
    ----------
    spectrogram: np.ndarray
        Spectrogram to be sonified.
    frequency_coefficients: np.ndarray, default = None
        Array containing frequency coefficients, in Hertz.
    time_coefficients: np.ndarray, default = None
        Array containing time coefficients, in seconds.
    sonification_duration: int, default = None
        Determines duration of sonification, in samples.
    fading_duration: float, default = 0.05
        Determines duration of fade-in and fade-out at beginning and end of the sonification, in seconds.
    normalize: bool, default = True
        Determines if output signal is normalized to [-1,1].
    fs: int, default = 22050
        Sampling rate, in samples per seconds.

    Returns
    -------
    spectrogram_sonification: np.ndarray
        Sonified spectrogram.
    """

    # Check if lengths of coefficient vectors match shape of spectrogram
    assert spectrogram.shape[0] == len(frequency_coefficients), f'The length of frequency_coefficients must match spectrogram.shape[0]'

    assert spectrogram.shape[1] == len(time_coefficients), f'The length of time_coefficients must match spectrogram.shape[1]'

    # Calculate Hop size from time_coefficients if not explicitly given
    H = int((time_coefficients[1] - time_coefficients[0]) * fs)

    # Determine length of sonification
    num_samples = sonification_duration if sonification_duration is not None else int((time_coefficients[-1] * fs) + H)

    # Initialize sonification
    spectrogram_sonification = np.zeros(num_samples)

    for i in range(spectrogram.shape[0]):
        weighting_vector = np.repeat(spectrogram[i, :], H)

        weighting_vector = smooth_weights(weights=weighting_vector, fading_samples=int(H / 8))

        sinusoid = generate_sinusoid(frequency=frequency_coefficients[i],
                                     phase=0,
                                     duration=num_samples / fs,
                                     fs=fs)

        spectrogram_sonification += (sinusoid * weighting_vector)

    spectrogram_sonification = fade_signal(spectrogram_sonification, fs=fs, fading_duration=fading_duration)

    spectrogram_sonification = normalize_signal(spectrogram_sonification) if normalize else spectrogram_sonification

    return spectrogram_sonification
