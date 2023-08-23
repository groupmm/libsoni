import numpy as np
import pandas as pd
import os
from libsoni.util.utils import generate_shepard_tone
from libsoni.util.utils import Hchord, add_to_sonification

VALID_STYLES = ['majmin', 'majmin_inv', 'shorthand', 'majminNumeric', 'extended']

#tests
#tests2
def sonify_chord_annotation(path_to_csv: str,
                            style: str = 'majmin',
                            filter: bool = False,
                            f_center: float = 440.0,
                            octave_cutoff: int = 1,
                            amplitude: float = 1.0,
                            f_tuning: float = 440,
                            fade_dur: float = 0.01,
                            fs: int = 44100
                            ) -> np.ndarray:
    """
        This function sonifies a chord annotation available as a .csv file.
        The chord_labels must be annotated according to the Harte convention 
        (Symbolic Representation of Musical Chords: a proposed syntax for text #annotations 2005 ISMIR)
        The sonification uses so-called Shepard tones, which can be shaped in their sound by using the 'filter' flag,
        as well as parameters f_center and octave_cutoff.

        Parameters
        ----------
        path_to_csv : str
            path to annotation file
        style : str, default: 'majmin'
            Annotation style of chord label
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

    chord_events_df = pd.read_csv(os.path.join(path_to_csv), delimiter=';', usecols=['start', 'end', style])

    y = np.zeros(np.ceil((chord_events_df.end.iloc[-1]) * fs).astype(int))

    for i, r in chord_events_df.iterrows():

        start, end, chord = r

        _, _, chord_as_chroma_vector = Hchord(chord).export_chroma_vectors()

        sonified_chord = np.zeros(int((end - start) * fs))
        # print(chord)
        for pitch_class, binary in enumerate(chord_as_chroma_vector):
            # print(pitch_class)
            if binary:
                sonified_chord += generate_shepard_tone(pitch_class=pitch_class,
                                                        filter=filter,
                                                        f_center=f_center,
                                                        octave_cutoff=octave_cutoff,
                                                        amplitude=amplitude,
                                                        duration=(end - start),
                                                        fs=fs,
                                                        f_tuning=f_tuning,
                                                        fade_dur=fade_dur)

        y = add_to_sonification(sonification=y, sonification_for_event=sonified_chord, start=start, fs=fs)

    return y
