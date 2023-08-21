import numpy as np


def sonify_f0(time_f0: np.ndarray,
              partials: np.ndarray = [1],
              partials_amplitudes: np.ndarray = [1],
              duration: int = None,
              fs: int = 22050) -> np.ndarray:
    # TODO: correct documentation
    """
    This function sonifies a F0 trajectory available as a .csv file.
    The sonification is related to the principle of a so-called numerical oscillator.
    The parameters harmonics and harmonics_amplitudes can be used to generate a desired sound through a specific overtone behavior.

    Parameters
    ----------
    time_f0 : str
        File path to the .csv file.
    partials : np.ndarray (default = [1])
        An array describing the number and frequency ratios of harmonics.
        An array [1] leads to sonification with only the fundamental frequency core,
        while an array [1,2] causes sonification with the fundamental frequency and twice the fundamental frequency.
    partials_amplitudes : np.ndarray (default = [1])
        Array that describes the amplitude ratios of harmonics.
        An array [1,0.5] causes the sinusoid with frequency core to have amplitude 1,
        while the sinusoid with frequency 2*core has amplitude 0.5.
    duration : int (default = None)
        duration of audio (given in samples)
    fs: int (default = 44100)
        Sampling rate (in Samples per second)

    Returns
    ----------
    y: np.ndarray
        Sonified core-trajectory.
    """
    times = time_f0[:, 0]
    f0s = time_f0[:, 1]

    num_samples = int(np.ceil(times[-1] * fs))
    f0s_sampled = np.zeros(num_samples)
    f0_sonification = np.zeros(num_samples)

    for i, (time, f0) in enumerate(zip(times, f0s)):
        if i == time_f0.shape[0] - 1:
            break
        next_time = times[i+1]
        f0s_sampled[int(time*fs):int(next_time*fs)] = f0

    for partial, partial_amplitude in zip(partials, partials_amplitudes):
        phase = 0
        phase_result = []
        for f0 in f0s_sampled:
            phase_step = 2 * np.pi * f0 * partial * 1 / fs
            phase += phase_step
            phase_result.append(phase)

        f0_sonification += np.sin(phase_result) * partial_amplitude

    if len(f0_sonification) <= duration:
        f0_sonification = f0_sonification
    

    return f0_sonification
