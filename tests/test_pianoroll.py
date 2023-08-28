import numpy as np
import pandas as pd
import os
from scipy.io import wavfile

from libsoni.core import pianoroll

FS = 22050
EXPORT_AUDIO = True
DURATIONS = [None, 4.2, 3.0, 5.0]


def test_pianoroll_clicks():
    pianoroll_df = pd.read_csv(os.path.join('data_csv', 'Bach_BWV1056-01-mm000-116_Oezer-V1_P_aligned.csv'),
                               delimiter=';')

    y = pianoroll.sonify_pianoroll_clicks(pianoroll_df,
                                          tuning_frequency=440.0,
                                          duration=None,
                                          fs=FS)

    if EXPORT_AUDIO:
        wavfile.write(os.path.join('tests', 'data_audio',
                                   'Bach_BWV1056-01-mm000-116_Oezer-V1_P_aligned_pianoroll_sonified_clicks.wav'), FS, y)


#test_pianoroll_clicks()

def test_pianoroll_additive_synthesis():
    pianoroll_df = pd.read_csv(os.path.join('data_csv', 'Bach_BWV1056-01-mm000-116_Oezer-V1_P_aligned.csv'),
                               delimiter=';')

    y = pianoroll.sonify_pianoroll_additive_synthesis(pianoroll_df,
                                                      partials=[1, 2, 3, 4],
                                                      partials_amplitudes=[1, 0.7, 0.3, 0.1],
                                                      partials_phase_offsets=[0, 0, 0, 0],
                                                      tuning_frequency=440.0,
                                                      duration=None,
                                                      fs=FS)

    if EXPORT_AUDIO:
        wavfile.write(os.path.join('tests', 'data_audio',
                                   'Bach_BWV1056-01-mm000-116_Oezer-V1_P_aligned_pianoroll_sonified_additive_synthesis.wav'), FS, y)


test_pianoroll_additive_synthesis()