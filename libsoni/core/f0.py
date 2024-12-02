import numpy as np

from ..utils import normalize_signal, fade_signal
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
              ignore_zero_freq_samples: int = 1000) -> np.ndarray:
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

    ignore_zero_freq_samples: int, default = 0
        Determines number of samples with frequency 0 will be ignored in sonification (e.g. not ideal f0-estimation). Must be greater than 2, otherwise ignored

    normalize: bool, default = True
        Determines if output signal is normalized to [-1,1].

    fs: int, default = 22050
        Sampling rate, in samples per seconds.

    Returns
    -------
    f0_sonification: np.ndarray (np.float32 / np.float64) [shape=(M, )]
        Sonified f0-trajectory.
    """
    if time_f0.ndim != 2:
        raise IndexError('time_f0 must be a numpy array of size [N, 2]')
    if time_f0.shape[1] != 2:
        raise IndexError('time_f0 must be a numpy array of size [N, 2]')

        

    if gains is not None:
        assert len(gains) == time_f0.shape[0], 'Array for confidence must have same length as time_f0.'
    else:
        gains = np.ones(time_f0.shape[0])
    time_positions = time_f0[:, 0]
    f0s = time_f0[:, 1]
    
    num_samples = int((time_positions[-1]) * fs)
    sample_positions = time_positions * fs
    sample_positions = sample_positions.astype(int)

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
   
    # Stretch f0s_stretched to match the given time positions.
    f0s_stretched = np.zeros(num_samples)
    gains_stretched = np.zeros(num_samples)
    for i, (time, f0, gain) in enumerate(zip(time_positions, f0s, gains)):
        if i == time_positions.shape[0] - 1:
            if not shorter_duration:
                f0s_stretched[int(time_positions[i] * fs):] = 0.0
                gains_stretched[int(time_positions[i] * fs):] = 0.0
        else:
            if(f0 < 0):
                f0 = 0
            next_time = time_positions[i + 1]
            f0s_stretched[int(time * fs):int(next_time * fs)] = f0
            gains_stretched[int(time * fs):int(next_time * fs)] = gain


    # Replace number of zero-frequencies with previous non-zerofreqency
    if(ignore_zero_freq_samples > 2 and num_samples > ignore_zero_freq_samples + 2 ):
        replaced_zeros = 0
        for i in range(num_samples- ignore_zero_freq_samples - 2) :
            if(f0s_stretched[i] == 0 and f0s_stretched[i-1] != 0):
                for j in range(ignore_zero_freq_samples):         
                    if (f0s_stretched[i+j] != 0):
                        break
                if(j == ignore_zero_freq_samples -1 and f0s_stretched[i+j] == 0):
                    continue
                else:
                    f0s_stretched[i:i+j] = f0s_stretched[i-1]
                    replaced_zeros += j
        
    
   


    # split f0 trajecotries into destinct notes (ratio between f0s > ±50 cent)
    splits = np.array([1])

    for i in range(num_samples-1):
        if(f0s_stretched[i+1] != 0):
            if(f0s_stretched[i]/f0s_stretched[i+1] > 2**(1/24) or f0s_stretched[i]/f0s_stretched[i+1] < 2**(-1/24) ):
                splits = np.append(splits, [int(i+1)])
        elif(f0s_stretched[i]!= 0):
            splits = np.append(splits, [int(i+1)])
    splits = np.delete(splits, 0)
    f0_sonification = np.zeros(len(f0s_stretched))
    
  

    
    
    # Sonification of individual Frequencieswith crossfades
    notes = np.split(f0s_stretched, splits)
    amps = np.split(gains_stretched, splits)
    cross_samples = int(crossfade_duration* fs)
    sample_start = 0
    sample_end = None
    
    for j in range(len(notes)):
        notes_current = notes[j]
        amps_current = amps[j] 

        sample_end = sample_start + int(len(notes_current))
        if(len(notes_current)< cross_samples):
            cross = int(len(notes_current)/2)
        else:
            cross = cross_samples
            
        if(j != 0):
            sample_end += cross
            notes_current = np.insert(notes_current,0,np.full(cross, notes_current[0]))
            amps_current = np.insert(amps_current,0,np.full(cross, amps_current[0]))
        
      
        if(np.mean(notes_current)> 0 ):
            signal =  generate_tone_instantaneous_phase(frequency_vector=notes_current,
                                                        gain_vector=amps_current,
                                                        partials=partials,
                                                        partials_amplitudes=partials_amplitudes,
                                                        partials_phase_offsets=partials_phase_offsets,
                                                        fading_duration=crossfade_duration,
                                                        fs=fs)
        
        
        else:
            signal = np.zeros(len(notes_current))

        
        f0_sonification[sample_start:sample_end] += signal
        sample_start = sample_end - cross
            

    f0_sonification = fade_signal(f0_sonification, fs=fs, fading_duration=fading_duration)
    f0_sonification = normalize_signal(f0_sonification) if normalize else f0_sonification

    return f0_sonification


