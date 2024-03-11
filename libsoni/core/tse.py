import numpy as np
from typing import List, Tuple

from libsoni.util.utils import normalize_signal, fade_signal
from libsoni.core.methods import generate_click


def sonify_tse_clicks(time_positions: np.ndarray = None,
                      click_pitch: int = 69,
                      click_fading_duration: float = 0.25,
                      click_amplitude: float = 1.0,
                      offset_relative: float = 0.0,
                      sonification_duration: int = None,
                      fading_duration: float = 0.05,
                      normalize: bool = True,
                      fs: int = 22050) -> np.ndarray:

    """Sonifies array of time positions with clicks.

    Parameters
    ----------
    time_positions: np.ndarray
        Array with time positions for clicks.

    click_pitch: int, default = 69
        Pitch for click signal.

    click_fading_duration: float, default = 0.25
        Fading duration for click signal, in seconds

    click_amplitude: float, default = 1.0
        Amplitude for click signal.

    offset_relative: float, default = 0.0
        Relative offset for the beginning of a click.
        0 indicates that the beginning of the click event is at the time position,
        1 indicates the ending of the click event corresponds to the time position.

    sonification_duration: int, default = None
        Determines duration of sonification, in samples.

    fading_duration: float, default = 0.05
        Determines duration of fade-in and fade-out at beginning and end of the sonification, in seconds.

    normalize: bool, default = True
        Determines if output signal is max-normalized to [-1,1].

    fs: int, default = 22050
        Sampling rate, in samples per seconds.

    Returns
    -------
    tse_sonification: np.ndarray
        Sonified time positions.
    """

    assert 0 <= click_pitch <= 127, f'Pitch is out of range [0,127].'
    assert 0 <= offset_relative <= 1, f'Relative offset is out of range [0,1].'

    num_samples = int((time_positions[-1] + click_fading_duration) * fs)

    if sonification_duration is None:
        sonification_duration = num_samples

    else:
        if sonification_duration < num_samples:
            duration_in_sec = sonification_duration / fs
            time_positions = time_positions[time_positions < duration_in_sec]
        num_samples = int(sonification_duration * fs)

    click = generate_click(pitch=click_pitch,
                           click_fading_duration=click_fading_duration,
                           amplitude=click_amplitude)

    tse_sonification = __sonify_tse_with_sound_event(click,
                                                     offset_relative,
                                                     time_positions,
                                                     num_samples,
                                                     fs,
                                                     fading_duration,
                                                     normalize)

    return tse_sonification[:sonification_duration]


def sonify_tse_sample(time_positions: np.ndarray = None,
                      sample: np.ndarray = None,
                      offset_relative: float = 0.0,
                      sonification_duration: int = None,
                      fading_duration: float = 0.05,
                      normalize: bool = True,
                      fs: int = 22050) -> np.ndarray:
    """Sonifies time positions with warped versions of a custom sample (e.g., metronome sounds).

    Parameters
    ----------
    time_positions: np.ndarray
        Array with time positions for clicks.

    sample: np.ndarray
        Sample to be used.

    offset_relative: float, default = 0.0
        Relative offset coefficient for the beginning of a click.
        0 indicates that the beginning of the click event is at the time position. 1 indicates the ending of the click
        event corresponds to the time position.

    sonification_duration: int, default = None
        Determines duration of sonification, in samples.

    fading_duration: float, default = 0.05
        Determines duration of fade-in and fade-out at beginning and end of the sonification, in seconds.

    normalize: bool, default = True
        Determines if output signal is max-normalized to [-1,1].

    fs: int, default = 22050
        Sampling rate, in samples per seconds.

    Returns
    -------
    tse_sonification: np.ndarray
        Sonified time positions.
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

    tse_sonification = __sonify_tse_with_sound_event(sample,
                                                     offset_relative,
                                                     time_positions,
                                                     num_samples,
                                                     fs,
                                                     fading_duration,
                                                     normalize)

    return tse_sonification[:sonification_duration]


def sonify_tse_multiple_clicks(times_pitches: List[Tuple[np.ndarray, int]] = None,
                               click_fading_duration: float = 0.25,
                               click_amplitude: float = 1.0,
                               offset_relative: float = 0.0,
                               sonification_duration: int = None,
                               fading_duration: float = 0.05,
                               normalize: bool = True,
                               fs: int = 22050) -> np.ndarray:
    """Given multiple arrays in form of a list, this function creates the sonification of different sources.

    Parameters
    ----------
    times_pitches: List[Tuple[np.ndarray, int]]
        List of tuples comprising the time positions and pitches of the clicks

    sonification_duration: int, default = None
        Duration of the output waveform, given in samples.

    click_fading_duration: float, default = 0.25
        Duration for click signal.

    click_amplitude: float, default = 1.0
        Amplitude for click signal.

    offset_relative: float, default = 0.0
        Relative offset coefficient for the beginning of the given audio sample.
        0 indicates that the beginning of the sample is at the time position. 1 indicates the ending of the sample
        corresponding to the time position.

    sonification_duration: int, default = None
        Determines duration of sonification, in samples.

    fading_duration: float, default = 0.05
        Determines duration of fade-in and fade-out at beginning and end of the sonification, in seconds.

    normalize: bool, default = True
        Determines if output signal is max-normalized to [-1,1].

    fs: int, default = 22050
        Sampling rate, in samples per seconds.

    Returns
    -------
    tse_sonification: np.ndarray
        Sonified time positions.
    """

    if sonification_duration is None:
        max_duration = 0
        for times_pitch in times_pitches:
            sonification_duration = times_pitch[0][-1]
            max_duration = sonification_duration if sonification_duration > max_duration else max_duration

        sonification_duration = int(np.ceil(fs * (max_duration + click_fading_duration)))
    tse_sonification = np.zeros(sonification_duration)

    for times_pitch in times_pitches:
        time_positions, pitch = times_pitch
        tse_sonification += sonify_tse_clicks(time_positions=time_positions,
                                              click_pitch=pitch,
                                              click_fading_duration=click_fading_duration,
                                              click_amplitude=click_amplitude,
                                              offset_relative=offset_relative,
                                              sonification_duration=sonification_duration,
                                              fs=fs)

    tse_sonification = fade_signal(tse_sonification, fs=fs, fading_duration=fading_duration)

    tse_sonification = normalize_signal(tse_sonification) if normalize else tse_sonification

    return tse_sonification


def sonify_tse_multiple_samples(times_samples: List[Tuple[np.ndarray, np.ndarray]] = None,
                                offset_relative: float = 0.0,
                                sonification_duration: int = None,
                                fading_duration: float = 0.05,
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

    sonification_duration: int, default = None
        Determines duration of sonification, in samples.

    fading_duration: float, default = 0.05
        Determines duration of fade-in and fade-out at beginning and end of the sonification, in seconds.

    normalize: bool, default = True
        Determines if output signal is max-normalized to [-1,1].

    fs: int, default = 22050
        Sampling rate, in samples per seconds.

    Returns
    -------
    tse_sonification: np.ndarray
        Sonified waveform in form of a 1D Numpy array.
    """

    if sonification_duration is None:
        max_duration = 0
        max_sample_duration_samples = 0

        for time_sample in times_samples:
            time_positions, sample = time_sample
            sonification_duration = time_positions[-1]
            duration_sample_samples = len(sample)
            max_duration = sonification_duration if sonification_duration > max_duration else max_duration
            max_sample_duration_samples = duration_sample_samples \
                if duration_sample_samples > max_sample_duration_samples else max_sample_duration_samples

        sonification_duration = int(np.ceil(fs * max_duration)) + max_sample_duration_samples

    tse_sonification = np.zeros(sonification_duration)

    for times_sample in times_samples:
        time_positions, sample = times_sample
        tse_sonification += sonify_tse_sample(time_positions=time_positions,
                                              sample=sample,
                                              offset_relative=offset_relative,
                                              sonification_duration=sonification_duration,
                                              fs=fs)

    tse_sonification = fade_signal(tse_sonification, fs=fs, fading_duration=fading_duration)
    tse_sonification = normalize_signal(tse_sonification) if normalize else tse_sonification

    return tse_sonification


def __sonify_tse_with_sound_event(sound_event: np.ndarray,
                                  offset_relative: float,
                                  time_positions: np.ndarray,
                                  num_samples: int,
                                  fs: int,
                                  fading_duration: float,
                                  normalize: bool) -> np.ndarray:
    """Sonify with sound events (e.g., beats, downbeats, etc.)

    Parameters
    ----------
    sound_event: np.ndarray
        A click signal or sample loaded from the disk

    offset_relative: float, default = 0.0
        Relative offset for the beginning of a sound event.
        0 indicates that the beginning of the sound event is at the time position,
        1 indicates the ending of the sound event corresponds to the time position.:

    time_positions: np.ndarray
        Array with time positions of the annotations.

    num_samples: int
        Number of samples of the output signals

    fs: int
        Sampling rate

    fading_duration: float
        Determines duration of fade-in and fade-out at beginning and end of the sonification, in seconds.

    normalize: bool, default = True
        Determines if output signal is max-normalized to [-1,1].

    Returns
    -------
    tse_sonification: np.ndarray
        Sonified signal with multiple sound events, given the time positions
    """

    tse_sonification = np.zeros(num_samples)
    num_click_samples = len(sound_event)
    offset_samples = int(offset_relative * num_click_samples)

    for idx, time_position in enumerate(time_positions):
        start_samples = int(time_position * fs) - offset_samples
        end_samples = start_samples + num_click_samples

        if start_samples < 0:
            if end_samples <= 0:
                continue
            tse_sonification[:end_samples] += sound_event[-end_samples:]

        else:
            tse_sonification[start_samples:end_samples] += sound_event

    tse_sonification = fade_signal(tse_sonification, fs=fs, fading_duration=fading_duration)
    tse_sonification = normalize_signal(tse_sonification) if normalize else tse_sonification

    return tse_sonification