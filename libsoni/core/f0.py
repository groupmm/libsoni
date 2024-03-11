import numpy as np

from libsoni.core.methods import generate_tone_instantaneous_phase
from libsoni.util.utils import normalize_signal, fade_signal


def sonify_f0(time_f0: np.ndarray,
              gains: np.ndarray = None,
              partials: np.ndarray = np.array([1]),
              partials_amplitudes: np.ndarray = np.array([1]),
              partials_phase_offsets: np.ndarray = None,
              sonification_duration: int = None,
              fading_duration: float = 0.05,
              normalize: bool = True,
              fs: int = 22050) -> np.ndarray:
    """Sonifies a F0 trajectory given as 2D Numpy array.

    The 2D array must contain time positions and the associated instantaneous frequencies.
    The sonification is based on the phase information by summation of the instantaneous frequencies.
    The parameters partials, partials_amplitudes and partials_phase_offsets can be used to shape the sound.

    Parameters
    ----------
    time_f0: np.ndarray
        2D array of time positions and f0s.

    gains: np.ndarray, default = None
        Array containing gain values for f0 values.

    partials: np.ndarray, default = [1]
        Array containing the desired partials of the fundamental frequencies for sonification.
        An array [1] leads to sonification with only the fundamental frequency,
        while an array [1,2] leads to sonification with the fundamental frequency and twice the fundamental frequency.

    partials_amplitudes: np.ndarray, default = None
        Array containing the amplitudes for partials.
        An array [1,0.5] causes the first partial to have amplitude 1,
        while the second partial has amplitude 0.5.
        If None, the amplitudes for all partials are set to 1.

    partials_phase_offsets: np.ndarray, default = None
        Array containing the phase offsets for partials.
        When not defined, the phase offsets for all partials are set to 0.

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
    f0_sonification: np.ndarray
        Sonified f0-trajectory.
    """
    if gains is not None:
        assert len(gains) == time_f0.shape[0], 'Array for confidence must have same length as time_f0.'
    else:
        gains = np.ones(time_f0.shape[0])
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

    f0s_stretched = np.zeros(num_samples)
    gains_stretched = np.zeros(num_samples)

    # Stretch f0s_stretched to match the given time positions.
    for i, (time, f0, gain) in enumerate(zip(time_positions, f0s, gains)):
        if i == time_positions.shape[0] - 1:
            if not shorter_duration:
                f0s_stretched[int(time_positions[i] * fs):] = 0.0
                gains_stretched[int(time_positions[i] * fs):] = 0.0
        else:
            next_time = time_positions[i + 1]
            f0s_stretched[int(time * fs):int(next_time * fs)] = f0
            gains_stretched[int(time * fs):int(next_time * fs)] = gain

    f0_sonification = generate_tone_instantaneous_phase(frequency_vector=f0s_stretched,
                                                        gain_vector=gains_stretched,
                                                        partials=partials,
                                                        partials_amplitudes=partials_amplitudes,
                                                        partials_phase_offsets=partials_phase_offsets,
                                                        fading_duration=fading_duration,
                                                        fs=fs)

    f0_sonification = fade_signal(f0_sonification, fs=fs, fading_duration=fading_duration)
    f0_sonification = normalize_signal(f0_sonification) if normalize else f0_sonification

    return f0_sonification
