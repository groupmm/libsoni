import numpy as np
from typing import List, Tuple

from libsoni.util.utils import generate_click, normalize_signal


def sonify_tse_click(time_positions: np.ndarray = None,
                     click_pitch: int = 69,
                     click_reverb_duration: float = 0.25,
                     click_amplitude: float = 1.0,
                     offset_relative: float = 0.0,
                     sonification_duration: int = None,
                     normalize: bool = True,
                     fs: int = 22050) -> np.ndarray:
    """This function sonifies an array containing time positions with clicks.

    Parameters
    ----------
    time_positions: np.ndarray
        Array with time positions for clicks.
    click_pitch: int, default = 69
        Pitch for click signal.
    click_reverb_duration: float, default = 0.25
        Duration for click signal.
    click_amplitude: float, default = 1.0
        Amplitude for click signal.
    offset_relative: float, default = 0.0
        Relative offset coefficient for the beginning of a click. 
        0 indicates that the beginning of the click event is at the time position. 1 indicates the ending of the click
        event corresponds to the time position. 
    sonification_duration: float, default = None
        Duration of the output waveform, given in samples.
    normalize: bool, default = True
        Decides, if output signal is normalized to [-1,1].
    fs: int, default = 22050
        Sampling rate

    Returns
    -------
    tse_sonification: np.ndarray
        Sonified waveform in form of a 1D Numpy array.

    """
    num_samples = int((time_positions[-1] + click_reverb_duration) * fs)
    if sonification_duration is None:
        sonification_duration = num_samples
    else:

        if sonification_duration < num_samples:
            duration_in_sec = sonification_duration / fs
            time_positions = time_positions[time_positions < duration_in_sec]

        num_samples = int(sonification_duration * fs)

    tse_sonification = np.zeros(num_samples)

    click = generate_click(pitch=click_pitch, reverb_duration=click_reverb_duration, amplitude=click_amplitude)

    num_click_samples = len(click)
    offset_samples = int(offset_relative * num_click_samples)
    for idx, time_position in enumerate(time_positions):

        start_samples = int(time_position * fs) - offset_samples
        end_samples = start_samples + num_click_samples
        if start_samples < 0:
            if end_samples <= 0:
                continue
            tse_sonification[:end_samples] += click[-end_samples:]
        else:
            tse_sonification[start_samples:end_samples] += click

    tse_sonification = normalize_signal(tse_sonification) if normalize else tse_sonification

    return tse_sonification[:sonification_duration]


def sonify_tse_sample(time_positions: np.ndarray = None,
                      sample: np.ndarray = None,
                      offset_relative: float = 0.0,
                      sonification_duration: int = None,
                      normalize: bool = True,
                      fs: int = 22050) -> np.ndarray:
    """This function sonifies an array containing time positions with clicks.

    Parameters
    ----------
    sample: np.ndarray
        Sample (e.g., recording of a kick drum)
    time_positions: np.ndarray
        Array with time positions for clicks.
    offset_relative: float, default = 0.0
        Relative offset coefficient for the beginning of the given audio sample.
        0 indicates that the beginning of the sample is at the time position. 1 indicates the ending of the sample
        corresponding to the time position.
    sonification_duration: float, default = None
        Duration of the output waveform, given in samples.
    normalize: bool, default = True
        Decides, if output signal is normalized to [-1,1].
    fs: int, default = 22050
        Sampling rate

    Returns
    -------
    tse_sonification: np.ndarray
        Sonified waveform in form of a 1D Numpy array.

    """
    sample_len = len(sample)
    num_samples = int((time_positions[-1]) * fs) + sample_len

    assert sample_len < time_positions[-1] * fs, f'The custom sample cannot be longer than the annotations.'
    if sonification_duration is not None:
        assert sample_len < sonification_duration, 'The custom sample cannot be longer than the sonification_duration.'

    if sonification_duration is None:
        sonification_duration = num_samples
    else:
        if sonification_duration < num_samples:
            duration_in_sec = sonification_duration / fs
            time_positions = time_positions[time_positions < duration_in_sec]

        num_samples = sonification_duration

    tse_sonification = np.zeros(num_samples)
    offset_samples = int(offset_relative * sample_len)
    for idx, time_position in enumerate(time_positions):
        start_samples = int(time_position * fs) - offset_samples
        end_samples = start_samples + sample_len

        if start_samples < 0:
            if end_samples <= 0:
                continue
            tse_sonification[:end_samples] += sample[-end_samples:]
        else:
            tse_sonification[start_samples:end_samples] += sample

    tse_sonification = normalize_signal(tse_sonification) if normalize else tse_sonification

    return tse_sonification[:sonification_duration]


def sonify_tse_multiple_clicks(times_pitches: List[Tuple[np.ndarray, int]] = None,
                               sonification_duration: int = None,
                               click_reverb_duration: float = 0.25,
                               click_amplitude: float = 1.0,
                               offset_relative: float = 0.0,
                               normalize: bool = True,
                               fs: int = 22050) -> np.ndarray:
    """Given multiple arrays in form of a list, this function creates the sonification of different sources.

    Parameters
    ----------
    times_pitches: List[Tuple[np.ndarray, int]]
        List of tuples comprising the time positions and pitches of the clicks
    sonification_duration: int
        Duration of the output waveform, given in samples.
    click_reverb_duration: float, default = 0.25
        Duration for click signal.
    click_amplitude: float, default = 1.0
        Amplitude for click signal.
    offset_relative: float, default = 0.0
        Relative offset coefficient for the beginning of the given audio sample.
        0 indicates that the beginning of the sample is at the time position. 1 indicates the ending of the sample
        corresponding to the time position.
    normalize: bool, default = True
        Decides, if output signal is normalized to [-1,1].
    fs: int, default = 22050
        Sampling rate
    Returns
    -------
    tse_sonification: np.ndarray
        Sonified waveform in form of a 1D Numpy array.
    """
    if sonification_duration is None:
        max_duration = 0
        for times_pitch in times_pitches:
            sonification_duration = times_pitch[0][-1]
            max_duration = sonification_duration if sonification_duration > max_duration else max_duration

        sonification_duration = int(np.ceil(fs * (max_duration + click_reverb_duration)))

    tse_sonification = np.zeros(sonification_duration)

    for times_pitch in times_pitches:
        time_positions, pitch = times_pitch

        tse_sonification += sonify_tse_click(time_positions=time_positions,
                                             click_pitch=pitch,
                                             click_reverb_duration=click_reverb_duration,
                                             click_amplitude=click_amplitude,
                                             offset_relative=offset_relative,
                                             sonification_duration=sonification_duration,
                                             fs=fs)

    tse_sonification = normalize_signal(tse_sonification) if normalize else tse_sonification

    return tse_sonification


def sonify_tse_multiple_samples(times_samples: List[Tuple[np.ndarray, np.ndarray]] = None,
                                offset_relative: float = 0.0,
                                duration: int = None,
                                normalize: bool = True,
                                fs: int = 22050) -> np.ndarray:
    """Given multiple arrays in form of a list, this function creates the sonification of different sources.

    Parameters
    ----------
    times_samples: List[Tuple[np.ndarray, np.ndarray]]
        List of tuples comprising the time positions and samples
    offset_relative: float, default = 0.0
        Relative offset coefficient for the beginning of the given audio sample.
        0 indicates that the beginning of the sample is at the time position. 1 indicates the ending of the sample
        corresponding to the time position.
    duration: int
        Duration of the output waveform, given in samples.
    normalize: bool, default = True
        Decides, if output signal is normalized to [-1,1].
    fs: int, default = 22050
        Sampling rate
    Returns
    -------
    tse_sonification: np.ndarray
        Sonified waveform in form of a 1D Numpy array.
    """

    if duration is None:
        max_duration = 0
        max_sample_duration_samples = 0
        for time_sample in times_samples:
            time_positions, sample = time_sample
            duration = time_positions[-1]
            duration_sample_samples = len(sample)
            max_duration = duration if duration > max_duration else max_duration
            max_sample_duration_samples = duration_sample_samples \
                if duration_sample_samples > max_sample_duration_samples else max_sample_duration_samples

        duration = int(np.ceil(fs * max_duration)) + max_sample_duration_samples

    tse_sonification = np.zeros(duration)

    for times_sample in times_samples:
        time_positions, sample = times_sample
        tse_sonification += sonify_tse_sample(time_positions=time_positions,
                                              sample=sample,
                                              offset_relative=offset_relative,
                                              sonification_duration=duration,
                                              fs=fs)

    tse_sonification = normalize_signal(tse_sonification) if normalize else tse_sonification

    return tse_sonification


def sonify_tse_text():
    # TODO: @yiitozer
    raise NotImplementedError
