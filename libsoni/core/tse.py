import pandas as pd
import numpy as np

from libsoni.util.utils import generate_click, load_sample, add_to_sonification


def sonify_tse_click(time_positions: np.ndarray = None,
                     click_pitch: int = 69,
                     click_duration: float = 0.25,
                     click_amplitude: float = 1.0,
                     offset_relative: float = 0.0,
                     duration: int = None,
                     fs: int = 22050):
    """This function sonifies an array containing time positions with clicks.
    Parameters
    ----------
    time_positions: np.ndarray
        Array with time positions for clicks.
    click_pitch: int, default = 69
        Pitch for click signal.
    click_duration: float, default = 0.25
        Duration for click signal.
    click_amplitude: float, default = 1.0
        amplitude for click signal.
    offset_relative: float, default = 0.0

    duration
    fs

    Returns
    -------

    """
    num_samples = int((time_positions[-1] + click_duration) * fs)
    if duration is None:
        duration = num_samples
    else:

        if duration < num_samples:
            duration_in_sec = duration / fs
            time_positions = time_positions[time_positions < duration_in_sec]

        num_samples = int(duration * fs)

    tse_sonification = np.zeros(num_samples)

    click = generate_click(pitch=click_pitch, duration=click_duration, amplitude=click_amplitude)

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

    return tse_sonification[:duration]


def sonify_tse_sample(time_positions: np.ndarray = None,
                      sample: np.ndarray = None,
                      offset_relative: float = 0.0,
                      duration: int = None,
                      fs: int = 22050):
    """This function sonifies an array containing time positions with clicks.
    Parameters
    ----------
    sample: np.ndarray
        Sample
    time_positions: np.ndarray
        Array with time positions for clicks.
    offset_relative: float, default = 0.0

    duration
    fs

    Returns
    -------

    """
    sample_len = len(sample)
    num_samples = int((time_positions[-1]) * fs) + sample_len

    assert sample_len < time_positions[-1] * fs, f'The custom sample cannot be longer than the annotations.'
    if duration is not None:
        assert sample_len < duration, 'The custom sample cannot be longer than the duration.'

    if duration is None:
        duration = num_samples
    else:
        if duration < num_samples:
            duration_in_sec = duration / fs
            time_positions = time_positions[time_positions < duration_in_sec]

        num_samples = int(duration * fs)

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

    return tse_sonification[:duration]


def sonify_tse_multiple_clicks(times_pitches: list = None,
                               duration: int = None,
                               click_duration: float = 0.25,
                               click_amplitude: float = 1.0,
                               offset_relative: float = 0.0,
                               fs: int = 22050) -> np.ndarray:
    if duration is None:
        max_duration = 0
        for times_pitch in times_pitches:
            duration = times_pitch[0][-1]
            max_duration = duration if duration > max_duration else max_duration

        duration = int(np.ceil(fs * (max_duration + click_duration)))

    tse_sonification = np.zeros(duration)

    for times_pitch in times_pitches:
        time_positions = times_pitch[0]
        pitch = times_pitch[1]

        tse_sonification += sonify_tse_click(time_positions=time_positions,
                                             click_pitch=pitch,
                                             click_duration=click_duration,
                                             click_amplitude=click_amplitude,
                                             offset_relative=offset_relative,
                                             duration=duration,
                                             fs=fs)

    # TODO: Check Normalization
    return tse_sonification


def sonify_tse_multiple_samples(times_samples: list = None,
                                offset_relative: float = 0.0,
                                duration: int = None,
                                fs: int = 22050):
    if duration is None:
        max_duration = 0
        max_sample_duration_samples = 0
        for time_sample in times_samples:
            duration = time_sample[0][-1]
            duration_sample_samples = len(time_sample[1])
            max_duration = duration if duration > max_duration else max_duration
            max_sample_duration_samples = duration_sample_samples if duration_sample_samples > max_sample_duration_samples else max_sample_duration_samples

        duration = int(np.ceil(fs * max_duration)) + max_sample_duration_samples

    tse_sonification = np.zeros(duration)

    for times_sample in times_samples:
        time_positions = times_sample[0]
        sample = times_sample[1]

        tse_sonification += sonify_tse_sample(time_positions=time_positions,
                                              sample=sample,
                                              offset_relative=offset_relative,
                                              duration=duration,
                                              fs=fs)

    return tse_sonification


def sonify_tse_text():
    # TODO: @yiitozer
    raise NotImplementedError
