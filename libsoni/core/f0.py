import numpy as np
from typing import Dict

from libsoni.util.utils import get_preset, normalize_signal


def sonify_f0(time_f0: np.ndarray,
              partials: np.ndarray = np.array([1]),
              partials_amplitudes: np.ndarray = np.array([1]),
              sonification_duration: int = None,
              normalize: bool = True,
              fs: int = 22050) -> np.ndarray:
    """This function sonifies a F0 trajectory from a 2D Numpy array.
    The sonification is related to the principle of a so-called numerical oscillator.
    The parameters partials and partials_amplitudes can be used to generate a desired sound through a specific
    overtone behavior.

    Parameters
    ----------
    time_f0: np.ndarray
        2D array of time positions and f0s
    partials: np.ndarray, default = [1]
        An array containing the desired partials of the fundamental frequencies for sonification.
            An array [1] leads to sonification with only the fundamental frequency core,
            while an array [1,2] causes sonification with the fundamental frequency and twice the fundamental frequency.
    partials_amplitudes: np.ndarray, default = [1]
        Array containing the amplitudes for partials.
            An array [1,0.5] causes the sinusoid with frequency core to have amplitude 1,
            while the sinusoid with frequency 2*core has amplitude 0.5.
    sonification_duration: int, default = None
        Duration of audio, given in samples
    normalize: bool, default = True
        Decides, if output signal is normalized to [-1,1].
    fs: int, default = 22050
        Sampling rate, in samples per seconds.

    Returns
    -------
    y: np.ndarray
        Sonified f0-trajectory.
    """
    # TODO: check for case, that time_positions are not monotonous (Sorting?)
    time_positions = time_f0[:, 0]
    f0s = time_f0[:, 1]
    num_samples = int(time_positions[-1] * fs)

    shorter_duration = False
    if sonification_duration is not None:
        duration_in_sec = sonification_duration / fs

        # if sonification_duration equals num_samples, do nothing
        if sonification_duration == num_samples:
            pass

        # if sonification_duration is less than num_samples, crop the arrays
        elif sonification_duration < num_samples:
            time_positions = time_positions[time_positions < duration_in_sec]
            time_positions = np.append(time_positions, duration_in_sec)
            f0s = f0s[:time_positions.shape[0]]
            shorter_duration = True
        # if sonification_duration is greater than num_samples, append
        else:
            time_positions = np.append(time_positions, duration_in_sec)
            f0s = np.append(f0s, 0.0)

        num_samples = int(time_positions[-1] * fs)

    f0s_streched = np.zeros(num_samples)
    f0_sonification = np.zeros(num_samples)

    # Strech f0s_streched to match the given time positions.
    for i, (time, f0) in enumerate(zip(time_positions, f0s)):
        if i == time_positions.shape[0] - 1:
            if not shorter_duration:
                f0s_streched[int(time_positions[i] * fs):] = 0.0
        else:
            next_time = time_positions[i+1]
            f0s_streched[int(time*fs):int(next_time*fs)] = f0

    #
    for partial, partial_amplitude in zip(partials, partials_amplitudes):
        phase = 0
        phase_result = []
        for f0 in f0s_streched:
            phase_step = 2 * np.pi * f0 * partial * 1 / fs
            phase += phase_step
            phase_result.append(phase)

        f0_sonification += np.sin(phase_result) * partial_amplitude

    f0_sonification = normalize_signal(f0_sonification) if normalize else f0_sonification

    return f0_sonification


def sonify_f0_with_presets(preset_dict: Dict = None,
                           sonification_duration: int = None,
                           normalize: bool = True,
                           fs: int = 22050) -> np.ndarray:
    """This function sonifies multiple f0 annotations with a certain preset.

    Parameters
    ----------
    preset_dict: dict
        Dictionary of presets in the following key-value pair format:
            {str: np.ndarray}
            preset: time_f0s
    sonification_duration: int, default = None
        Duration of the output waveform, given in samples
    normalize: bool, default = True
        Decides, if output signal is normalized to [-1,1].
    fs: int
        Sampling rate

    Returns
    -------
    f0_sonification: np.ndarray
        Sonified waveform
    """
    if sonification_duration is None:
        max_duration = 0
        for label in preset_dict:
            sonification_duration = preset_dict[label]['time_f0'][-1, 0]
            max_duration = sonification_duration if sonification_duration > max_duration else max_duration
        sonification_duration = int(np.ceil(fs * max_duration))

    f0_sonification = np.zeros(sonification_duration)

    for label in preset_dict:
        preset_features_dict = get_preset(preset_dict[label]['preset'])
        gain = preset_dict[label]['gain'] if 'gain' in preset_dict[label] else 1

        f0_sonification += sonify_f0(time_f0=preset_dict[label]['time_f0'],
                                     partials=preset_features_dict['partials'],
                                     partials_amplitudes=preset_features_dict['amplitudes'],
                                     sonification_duration=sonification_duration,
                                     fs=fs) * gain

    f0_sonification = normalize_signal(f0_sonification) if normalize else f0_sonification

    return f0_sonification
