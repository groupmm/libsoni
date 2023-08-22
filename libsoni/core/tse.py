import pandas as pd
import numpy as np
from decimal import Decimal, ROUND_DOWN

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
        duration_in_sec = duration / fs

        if duration < num_samples:
            time_positions = time_positions[time_positions < duration_in_sec]
            num_samples = int((time_positions[-1] + click_duration) * fs)

    tse_sonification = np.zeros(num_samples)

    click = generate_click(pitch=click_pitch, duration=click_duration, amplitude=click_amplitude)
    for idx, time_position in enumerate(time_positions):
        start_sec = time_position - offset_relative * click_duration

        if start_sec < 0:
            end_sec = start_sec + click_duration
            end_samples = int(fs * end_sec)
            if end_sec <= 0:
                continue
            tse_sonification[:end_samples] += click[-end_samples:]
        else:
            start_samples = int(start_sec * fs)
            end_samples = start_samples + len(click)
            tse_sonification[start_samples:end_samples] += click

    return tse_sonification[:duration]


###########################
def sonify_beat_annotation(path_to_csv: str,
                           sonification_method: str = 'click',
                           pitch_downbeat: int = 81,
                           pitch_upbeat: int = 69,
                           downbeat_sample: str = '',
                           upbeat_sample: str = '',
                           amplitude_downbeat: float = 1.0,
                           amplitude_upbeat: float = 1.0,
                           duration: float = 0.2,
                           fs: int = 44100) -> np.ndarray:
    """
    This function sonifies the entries of a beat annotation in .csv format. (see /docs/annotation_conventions.txt for more information)

    Parameters
    ----------
    path_to_csv : str
        path of the annotation file
    sonification_method : str, default: 'click'
        sonification method, either 'click' or 'sample'
    pitch_downbeat : int, default: 81
        pitch for the downbeat click signal
    pitch_upbeat : int, default: 69
        pitch for the upbeat click signal
    downbeat_sample : str, default: ''
        path to the desired downbeat sample
    upbeat_sample : str, default: ''
        path to the desired upbeat sample
    amplitude_downbeat : float, default: 1.0
        amplitude for the downbeat click signal
    amplitude_upbeat : float, default: 1.0
        amplitude for the upbeat click signal
    duration : float, default. 0.2
        duration (in seconds) for downbeat and upbeat signal
    fs : int, default: 44100
        Sampling rate (in Samples per second)

    Returns
    ----------
    y: array-like
        Sonified beat annotation
    """

    # read annotation file
    beat_events_df = pd.read_csv(path_to_csv, delimiter=';')

    # create empty array according to the time bounds given by the annotation file.
    y = np.zeros(np.ceil((max(beat_events_df.start.unique() + duration) * fs)).astype(int))

    if sonification_method == 'click':

        # create click-signals for downbeat and upbeat events
        downbeat_signal = click(pitch=pitch_downbeat, amplitude=amplitude_downbeat, duration=duration, fs=fs)
        upbeat_signal = click(pitch=pitch_upbeat, amplitude=amplitude_upbeat, duration=duration, fs=fs)

    elif sonification_method == 'sample':
        # load samples for downbeat and upbeat events
        downbeat_signal, _ = load_sample(downbeat_sample, fs)
        upbeat_signal, _ = load_sample(upbeat_sample, fs)

    # iterate beat events of the annotation file and insert corresponding signals at the corresponding temporal positions
    for i, r in beat_events_df.iterrows():
        start, beat = r
        beat = Decimal(str(beat)).quantize(Decimal('0.000'), rounding=ROUND_DOWN)

        # check if beat is downbeat or upbeat (see docs for more information)
        if str(beat)[-3:] == '000' or str(beat)[-1] == '1' or beat == 1:
            # add downbeat_signal to sonification
            y = add_to_sonification(sonification=y, sonification_for_event=downbeat_signal, start=start, fs=fs)
        else:
            # add upbeat_signal to sonification
            y = add_to_sonification(sonification=y, sonification_for_event=upbeat_signal, start=start, fs=fs)

    return y
