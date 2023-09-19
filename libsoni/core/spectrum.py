import numpy as np
from typing import Tuple

from libsoni.util.utils import normalize_signal, fade_signal, smooth_weights
from libsoni.core.methods import generate_sinusoid


def sonify_spectrum(spectogram: np.ndarray,
                    frequency_coef: np.ndarray,
                    time_coef: np.ndarray,
                    fade_duration: float = 0.05,
                    sonification_duration: float = None,
                    normalize: bool = True,
                    fs: int = 22050,
                    tuning_frequency: float = 440.0) -> np.ndarray:
    """Sonify chromagram

        Parameters
        ----------
        spectogram: np.ndarray
            Magnitude spectogram to be sonified.
        frequency_coef: np.ndarray
            Array containing frequencies for sonification.
        time_coef: np.ndarray,
            Array containing time coefficients, in seconds, for sonification.
        sonification_duration: int, default = None
            Duration of audio, given in samples
        fade_duration: float, default = 0.05
            Duration of fade-in and fade-out at beginning and end of the sonification, given in seconds.
        fs: int, default: 44100
            sampling rate in Samples/second
        normalize: bool, default = True
            Decides, if output signal is normalized to [-1,1].
        fs: int, default = 22050
            Sampling rate, in samples per seconds.
        tuning_frequency : float, default = 440
            Tuning frequency.

        Returns
        -------
            y: synthesized tone
        """
    assert spectogram.shape[0] == len(frequency_coef), f'The length of frequency_coef must match spectogram.shape[0]'
    assert spectogram.shape[1] == len(time_coef), f'The length of time_coef must match spectogram.shape[1]'

    # Calculate hop size from time coefs
    H = int((time_coef[1] - time_coef[0]) * fs)
    # Determine length of sonification
    num_samples = sonification_duration if sonification_duration is not None else int(time_coef[-1] * fs + H)
    # Initialize sonification
    spectogram_sonification = np.zeros(num_samples)

    for i in range(spectogram.shape[0]):

        weighting_vector = np.repeat(spectogram[i, :], int((time_coef[1] - time_coef[0]) * fs))

        weighting_vector_smoothed = smooth_weights(weights=weighting_vector,
                                                   fading_samples=8)
        # TODO: Fading samples?
        sinusoid = generate_sinusoid(frequency=frequency_coef[i],
                                     phase=0,
                                     num_samples=num_samples,
                                     fs=fs)

        spectogram_sonification += (sinusoid * weighting_vector_smoothed)
    spectogram_sonification = fade_signal(spectogram_sonification, fs=fs, fading_sec=fade_duration)

    spectogram_sonification = normalize_signal(spectogram_sonification) if normalize else spectogram_sonification

    return spectogram_sonification