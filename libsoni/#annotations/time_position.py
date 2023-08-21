import numpy as np
import pandas as pd
import os
from libsoni.util.utils import click, load_sample, add_to_sonification

# TODO : merge functions
def sonify_time_position_annotation_with_clicks(path_to_csv: str, pitch: int = 69, amplitude: float = 1.0, duration: float = 0.2, fs: int = 44100) -> np.ndarray:
    """
    This function sonifies the entries of a time position annotation in .csv format using 'click signals'. (see /docs/annotation_conventions.txt for more information)

    Parameters
    ----------
    path_to_csv : str
        path to annotation file
    pitch : int
        pitch of click
    amplitude : float
        amplitude of click
    duration : float
        duration of click
    fs : int
        sampling rate
    Returns
    ----------
    y: array-like
        Sonification of time position annotation
    """

    # read annotation file
    time_position_annotation_df = pd.read_csv(os.path.join(path_to_csv), delimiter=';')

    # create empty array according to the time bounds given by the annotation file
    y = np.zeros(np.ceil((time_position_annotation_df.start.iloc[-1] + duration) * fs).astype(int))

    # create click-signal for time position events
    time_position_click = click(pitch=pitch, amplitude=amplitude, duration=duration, fs=fs)

    for i, r in time_position_annotation_df.iterrows():
        start = r

        # add measure_click to sonification
        y = add_to_sonification(sonification=y, sonification_for_event=time_position_click, start=start, fs=fs)

    return y


def sonify_time_position_annotation_with_sample(path_to_csv: str, sample: str, fs: int = 44100) -> np.ndarray:
    """
    This function sonifies the entries of a time position annotation in .csv format using audio samples. (see /docs/annotation_conventions.txt for more information)

    Parameters
    ----------
    path_to_csv : str
        path of the annotation file
    sample : str
        sample for time position event
    fs : int
        sampling rate

    Returns
    ----------
    y: array-like
        sonification
    """

    # read annotation file
    time_position_annotation_df = pd.read_csv(os.path.join(path_to_csv), delimiter=';')

    # create empty array according to the time bounds given by the annotation file
    y = np.zeros(np.floor(time_position_annotation_df.start.iloc[-1] + len(sample)) * fs).astype(int)

    # load sample for time position event
    sample = load_sample(sample)

    # iterate measure events of the annotation file and insert corresponding click signals at the corresponding temporal positions
    for i, r in time_position_annotation_df.iterrows():
        start, _ = r

        # add sample to sonification
        y = add_to_sonification(sonification=y, sonification_for_event=sample, start=start, fs=fs)

    return y
