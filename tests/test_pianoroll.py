import numpy as np
import pandas as pd
import soundfile as sf

from libsoni.core.pianoroll import sonify_pianoroll_clicks,\
    sonify_pianoroll_additive_synthesis, sonify_pianoroll_fm_synthesis, sonify_pianoroll_sample

Fs = 22050
SAMPLE, _ = sf.read('data_audio/samples/01Pia1F060f_np___0_short.wav', Fs)
DF_PIANOROLL = pd.read_csv('data_csv/demo_pianoroll/FMP_B_Sonify_Pitch_Schubert_D911-11_SC06.csv',
                           sep=';')
PARTIALS = [np.array([1]), np.array([1, 2, 3])]
PARTIALS_AMPLITUDES = [np.array([1]), np.array([1, 0.5, 0.25])]


def test_pianoroll_additive():
    for partial_idx, partials in enumerate(PARTIALS):
        y = sonify_pianoroll_additive_synthesis(pianoroll_df=DF_PIANOROLL,
                                                partials=partials,
                                                partials_amplitudes=PARTIALS_AMPLITUDES[partial_idx],
                                                sonification_duration=5.0)

        ref, _ = sf.read(f'tests/data/pianoroll_add_{partial_idx}.wav')
        assert np.allclose(y, ref, atol=1e-4, rtol=1e-5)


def test_pianoroll_sample():
    y = sonify_pianoroll_sample(pianoroll_df=DF_PIANOROLL,
                                sample=SAMPLE,
                                reference_pitch=60,
                                sonification_duration=3.0)
    ref, _ = sf.read(f'tests/data/pianoroll_sample.wav')
    assert np.allclose(y, ref, atol=1e-4, rtol=1e-5)


def test_pianoroll_fm():
    y = sonify_pianoroll_fm_synthesis(pianoroll_df=DF_PIANOROLL,
                                      sonification_duration=3.0,
                                      mod_rate_relative=0.5,
                                      mod_amp=0.1)

    ref, _ = sf.read(f'tests/data/pianoroll_fm.wav')
    assert np.allclose(y, ref, atol=1e-4, rtol=1e-5)


def test_pianoroll_clicks():
    y = sonify_pianoroll_clicks(pianoroll_df=DF_PIANOROLL,
                                sonification_duration=3.0)

    ref, _ = sf.read(f'tests/data/pianoroll_clicks.wav')
    assert np.allclose(y, ref, atol=1e-4, rtol=1e-5)
