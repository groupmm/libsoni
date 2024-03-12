import numpy as np
import pandas as pd
import soundfile as sf

from libsoni.core.pianoroll import sonify_pianoroll_clicks,\
    sonify_pianoroll_additive_synthesis, sonify_pianoroll_fm_synthesis, sonify_pianoroll_sample

Fs = 22050
SAMPLE, _ = sf.read('data_audio/samples/01Pia1F060f_np___0_short.wav', Fs)

def test_pianoroll_sample():
    pass

def test_pianoroll_fm():
    pass

def test_pianoroll_additive():

def test_pianoroll_clicks():
    pass
