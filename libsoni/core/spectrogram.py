import numpy as np
from libsoni.util.utils import normalize_signal, fade_signal, smooth_weights
from libsoni.core.methods import generate_sinusoid
from concurrent.futures import ProcessPoolExecutor
import os


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
    assert spectrogram.shape[0] == len(frequency_coefficients),\
        f'The length of frequency_coefficients must match spectrogram.shape[0]'

    assert spectrogram.shape[1] == len(time_coefficients), \
        f'The length of time_coefficients must match spectrogram.shape[1]'

    # Calculate Hop size from time_coefficients if not explicitly given
    H = int((time_coefficients[1] - time_coefficients[0]) * fs)

    # Determine length of sonification
    num_samples = sonification_duration if sonification_duration is not None else int(np.ceil(time_coefficients[-1] * fs) + H)

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


def sonify_spectrogram_multi(spectrogram: np.ndarray,
                             frequency_coefficients: np.ndarray = None,
                             time_coefficients: np.ndarray = None,
                             sonification_duration: int = None,
                             fading_duration: float = 0.05,
                             fs: int = 22050,
                             num_processes: int = None) -> np.ndarray:
    """Sonifies a spectrogram using sinusoids, using multiprocessing for efficiency.

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

    fs: int, default = 22050
        Sampling rate, in samples per seconds.

    num_processes: int, default = None
        Number of processes
    Returns
    -------
    spectrogram_sonification: np.ndarray
        Sonified spectrogram.
    """

    assert spectrogram.shape[0] == len(frequency_coefficients), \
        f'The length of frequency_coefficients must match spectrogram.shape[0]'
    assert spectrogram.shape[1] == len(time_coefficients), \
        f'The length of time_coefficients must match spectrogram.shape[1]'

    if num_processes is None:
        num_processes = os.cpu_count() or 1

    H = int(np.ceil((time_coefficients[1] - time_coefficients[0]) * fs))
    num_samples = sonification_duration if sonification_duration is not None else int(np.ceil(time_coefficients[-1] * fs) + H)

    spectrogram_sonification = np.zeros(num_samples, dtype=np.float64)

    num_processes = min(num_processes, spectrogram.shape[0])

    with ProcessPoolExecutor(max_workers=num_processes) as executor:
        chunk_size = spectrogram.shape[0] // num_processes
        args_list = [
            (
                i * chunk_size,
                min((i + 1) * chunk_size, spectrogram.shape[0]),
                spectrogram[i * chunk_size: min((i + 1) * chunk_size, spectrogram.shape[0]), :],
                frequency_coefficients[i * chunk_size: min((i + 1) * chunk_size, spectrogram.shape[0])],
                time_coefficients,
                num_samples,
                H,
                fs
            )
            for i in range(num_processes)
        ]
        results = list(executor.map(__sonify_chunk, args_list))
    for result in results:
        spectrogram_sonification += result

    spectrogram_sonification = fade_signal(spectrogram_sonification, fs=fs, fading_duration=fading_duration)
    spectrogram_sonification /= np.max(spectrogram_sonification)

    return spectrogram_sonification

def __sonify_chunk(args):
    start, end, spectrogram_chunk, frequency_coefficients_chunk, time_coefficients, num_samples, H, fs = args

    spectrogram_sonification_chunk = np.zeros(num_samples)

    for i in range(spectrogram_chunk.shape[0]):
        weighting_vector = np.repeat(spectrogram_chunk[i, :], H)

        weighting_vector = smooth_weights(weights=weighting_vector, fading_samples=int(H / 8))

        sinusoid = generate_sinusoid(frequency=frequency_coefficients_chunk[i],
                                     phase=0,
                                     duration=(len(weighting_vector)/fs),
                                     fading_duration=0.05,
                                     fs=fs)

        spectrogram_sonification_chunk += (sinusoid * weighting_vector)
    return spectrogram_sonification_chunk

