import numpy as np

from ..utils import normalize_signal, fade_signal, split_freq_trajectory, replace_zeros
from .methods import generate_tone_instantaneous_phase


def sonify_f0(time_f0: np.ndarray,
              gains: np.ndarray = None,
              partials: np.ndarray = np.array([1]),
              partials_amplitudes: np.ndarray = np.array([1]),
              partials_phase_offsets: np.ndarray = None,
              sonification_duration: int = None,
              fading_duration: float = 0.05,
              crossfade_duration = 0.05,
              normalize: bool = True,
              fs: int = 22050,
              ignore_zero_freq_samples: int = 1000,
              freq_change_threshold_cents: float = 50) -> np.ndarray:
    """Sonifies a F0 trajectory given as 2D Numpy array.

    The 2D array must contain time positions and the associated instantaneous frequencies.
    The sonification is based on the phase information by summation of the instantaneous frequencies.
    The parameters partials, partials_amplitudes and partials_phase_offsets can be used to shape the sound.

    Parameters
    ----------
    time_f0: np.ndarray  (np.float32 / np.float64) [shape=(N, 2)]
        2D array of time positions and f0s.

    gains: np.ndarray (np.float32 / np.float64) [shape=(N, )], default = None
        Array containing gain values for f0 values.

    partials: np.ndarray (np.float32 / np.float64) [shape=(N, )], default = [1]
        Array containing the desired partials of the fundamental frequencies for sonification.
        An array [1] leads to sonification with only the fundamental frequency,
        while an array [1,2] leads to sonification with the fundamental frequency and twice the fundamental frequency.

    partials_amplitudes: np.ndarray (np.float32 / np.float64) [shape=(N, )], default = None
        Array containing the amplitudes for partials.
        An array [1,0.5] causes the first partial to have amplitude 1,
        while the second partial has amplitude 0.5.
        If None, the amplitudes for all partials are set to 1.

    partials_phase_offsets: np.ndarray (np.float32 / np.float64) [shape=(N, )], default = None
        Array containing the phase offsets for partials.
        When not defined, the phase offsets for all partials are set to 0.

    sonification_duration: int, default = None
        Determines duration of sonification, in samples.

    fading_duration: float, default = 0.05
        Determines duration of fade-in and fade-out at beginning and end of the sonification, in seconds.

    crossfade_duration: float, default = 0.05
        Determines duration of crossfade between two destinct frequency-samples (±50 cents), in seconds.

    normalize: bool, default = True
        Determines if output signal is normalized to [-1,1].

    fs: int, default = 22050
        Sampling rate, in samples per seconds.

    ignore_zero_freq_samples: int, default = 1000
        Determines number of samples with frequency 0 will be ignored in sonification (e.g. not ideal f0-estimation).
        Must be greater than 2, otherwise ignored.

    freq_change_threshold_cents: float, default = 50
        If the change in frequency between successive frames is larger than this threshold in cents, the sonification
        will be crossfaded instead of a linear interpolation of the instantaneous frequency

    Returns
    -------
    f0_sonification: np.ndarray (np.float32 / np.float64) [shape=(M, )]
        Sonified f0-trajectory.
    """
    if time_f0.ndim != 2 or time_f0.shape[1] != 2:
        raise IndexError('time_f0 must be a numpy array of size [N, 2]')

    if gains is not None:
        assert len(gains) == time_f0.shape[0], 'Array for confidence must have same length as time_f0.'
    else:
        gains = np.ones(time_f0.shape[0])
    time_positions = time_f0[:, 0]
    f0s = time_f0[:, 1]
    
    num_samples = int((time_positions[-1]) * fs)
    sample_positions = (time_positions * fs).astype(int)

    # crop or expand given time/F0 arrays if a desired sonification duration is given
    shorter_duration = False
    if sonification_duration is not None:
        duration_in_sec = sonification_duration / fs

        if sonification_duration == num_samples:
            pass
        elif sonification_duration < num_samples:
            # crop the time/F0 array
            time_positions = time_positions[time_positions < duration_in_sec]
            time_positions = np.append(time_positions, duration_in_sec)
            f0s = f0s[:time_positions.shape[0]]
            shorter_duration = True
        else: # sonification_duration > num_samples
            # expand the time/F0 array with frequency 0 at last time position
            time_positions = np.append(time_positions, duration_in_sec)
            f0s = np.append(f0s, 0.0)

        num_samples = int(time_positions[-1] * fs)
   
    # stretch F0 to instantaneous frequency per sample
    f0_inst = np.zeros(num_samples)
    gains_inst = np.zeros(num_samples)
    for i, (time, f0, gain) in enumerate(zip(time_positions, f0s, gains)):
        if i == time_positions.shape[0] - 1:
            if not shorter_duration:
                f0_inst[int(time_positions[i] * fs):] = 0.0
                gains_inst[int(time_positions[i] * fs):] = 0.0
        else:
            if(f0 < 0):
                f0 = 0
            next_time = time_positions[i + 1]
            f0_inst[int(time * fs):int(next_time * fs)] = f0
            gains_inst[int(time * fs):int(next_time * fs)] = gain


    # replace short zero-frequency segments with previous non-zero freqency to avoid audible artifacts
    f0_inst = replace_zeros(f0_inst, ignore_zero_freq_samples)

    # split F0 trajectories into separate regions in which the frequency change is within a threshold
    # sonification will be cross-faded between regions
    splits = split_freq_trajectory(f0_inst, freq_change_threshold_cents)
    notes = np.split(f0_inst, splits)
    amps = np.split(gains_inst, splits)
  
    # sonification of individual regions with crossfades
    N_fade = int(crossfade_duration * fs)
    N_fade_in = N_fade
    sample_start = 0
    sample_end = None
    f0_sonification = np.zeros(num_samples)

    
    for j in range(len(notes)):
        notes_current = notes[j]
        amps_current = amps[j] 

        sample_end = sample_start + len(notes_current)
        if len(notes_current) < N_fade:
            N_fade_out = int(len(notes_current))
        else:
            N_fade_out = N_fade

        if j == 0 and len(notes_current) < N_fade_in:
            N_fade_in = int(len(notes_current))

        # extend note in the beginning for a smooth crossfade
        if j != 0:
            sample_end += N_fade_in
            notes_current = np.pad(notes_current, (N_fade_in, 0), mode="edge")
            amps_current = np.pad(amps_current, (N_fade_in, 0), mode="edge")
        

        if np.any(notes_current > 0):
            signal =  generate_tone_instantaneous_phase(frequency_vector=notes_current,
                                                        gain_vector=amps_current,
                                                        partials=partials,
                                                        partials_amplitudes=partials_amplitudes,
                                                        partials_phase_offsets=partials_phase_offsets,
                                                        fading_duration=(N_fade_in/fs, N_fade_out/fs),
                                                        fs=fs)
        
        
        else:
            # if all frequencies are zero, do not call generate function to avoid DC offset
            signal = np.zeros(len(notes_current))

        print(sample_start / fs, sample_end / fs, notes_current[-1])
        print(N_fade_in / fs, N_fade_out / fs)
        f0_sonification[sample_start:sample_end] += signal
        N_fade_in = N_fade_out
        sample_start = sample_end - N_fade_in
            

    f0_sonification = normalize_signal(f0_sonification) if normalize else f0_sonification

    return f0_sonification


