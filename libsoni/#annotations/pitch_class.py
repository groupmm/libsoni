import numpy as np
import pandas as pd
import os
from libsoni.util.utils import generate_shepard_tone, add_to_sonification


def sonify_pitchclass_annotation(path_to_csv: str,
                                 filter: bool = False,
                                 f_center: float = 440.0,
                                 octave_cutoff: int = 1,
                                 amplitude: float = 1.0,
                                 f_tuning: float = 440,
                                 fade_dur: float = 0.01,
                                 fs: int = 44100
                                 ) -> np.ndarray:
    """
        This function sonifies a pitch class annotation available as a .csv file.

        Parameters
        ----------
        path_to_csv : str
            path to annotation file
        filter : bool, default: False
            Decides, if Shepard tones are filtered or not
        f_center : float, default: 440.0
            Center frequency (in Hertz) for bell-shaped filter
        octave_cutoff : int, default: 1
            Determines, at which multiple of f_center, the harmonics get attenuated by 2.
        amplitude : float, default: 1.0
            Amplitude of resulting signal
        f_tuning : float, default: 440.0
            Tuning frequency (in Hertz)
        fade_dur : float, default = 0.01
            Duration (in seconds) of fade in and fade out
        fs : int, default: 44100
            Sampling rate (in Samples per second)

        Returns
        -------
        y: np.ndarray
            Sonified chord annotation
    """

    pitchclass_df = pd.read_csv(os.path.join(path_to_csv), delimiter=';')

    y = np.zeros(np.ceil(max(pitchclass_df.end.unique()) * fs).astype(int))

    for i, r in pitchclass_df.iterrows():
        start, end, pitchclass = r
        pitchclass_signal = np.zeros(int((end - start) * fs))
        pitchclass_signal += generate_shepard_tone(pitch_class=pitchclass,
                                                   filter=filter,
                                                   f_center=f_center,
                                                   octave_cutoff=octave_cutoff,
                                                   amplitude=amplitude,
                                                   duration=(end - start),
                                                   fs=fs,
                                                   f_tuning=f_tuning,
                                                   fade_dur=fade_dur)

        y = add_to_sonification(sonification=y, sonification_for_event=pitchclass_signal, start=start, fs=fs)
    return y
